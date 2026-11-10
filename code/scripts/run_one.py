"""Train (if learned) and evaluate one method / system / seed.  Writes results/<out>/<method>_<system>_s<seed>.json (+ checkpoint).
TEST pools are refused unless DSR_ALLOW_TEST=1 (set only by the frozen final pipeline)."""
import argparse, json, os, pickle, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsr.env import DSREnv, Scenario
from agents.common import Runner, evaluate, summarize
from agents.registry import LEARNED, PLANNERS, PlannerAgent

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def load_pools(system):
    raw = pickle.load(open(os.path.join(ROOT, "data", f"pools_{system}.pkl"), "rb"))
    pools = {k: [Scenario(f, d, h, s, sp) for (f, d, h, s, sp) in v] for k, v in raw.items()}
    excl = os.environ.get("DSR_EXCLUDE_ZONE")
    if excl not in (None, ""):          # leave-zone-out: validation pools used for checkpoint selection also exclude the held-out zone
        from dsr.network import build
        n = build(system)
        for k in ("val_select", "val_dev"):
            pools[k] = [s for s in pools[k] if all(n.zone[n.t[l]] != int(excl) for l in s.faults)]
    return pools


def count_params(agent):
    for attr in ("actors", "net", "q", "mods"):
        if hasattr(agent, attr):
            mod = getattr(agent, attr)
            if hasattr(mod, "parameters"):
                return sum(p.numel() for p in mod.parameters())
    return 0


def total_params(agent):
    tot = 0
    for attr in ("actors", "critics", "net", "q", "mods"):
        if hasattr(agent, attr):
            tot += sum(p.numel() for p in getattr(agent, attr).parameters())
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--system", default="ieee33"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2000); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=None); ap.add_argument("--min_episodes", type=int, default=0)
    ap.add_argument("--pools", default="val_dev"); ap.add_argument("--out", default="results/screening")
    ap.add_argument("--cfg", default="{}"); ap.add_argument("--wall", type=float, default=None); ap.add_argument("--tag", default="")
    ap.add_argument("--n_val", type=int, default=60); ap.add_argument("--time_decisions", action="store_true")
    ap.add_argument("--save_ckpt", action="store_true"); ap.add_argument("--lookahead", default=""); ap.add_argument("--init_from", default="")
    a = ap.parse_args()
    pools_wanted = a.pools.split(",")
    if any(p.startswith("test") for p in pools_wanted) and os.environ.get("DSR_ALLOW_TEST") != "1":
        sys.exit("refusing to evaluate TEST pools outside the frozen final pipeline (set DSR_ALLOW_TEST=1)")
    cfg = json.loads(a.cfg)
    pools = load_pools(a.system)
    outdir = os.path.join(ROOT, a.out); os.makedirs(outdir, exist_ok=True)
    name = f"{a.method}{a.tag}_{a.system}_s{a.seed}"
    t0 = time.time()
    result = dict(method=a.method, system=a.system, seed=a.seed, cfg=cfg, episodes_budget=a.episodes)
    if a.method in PLANNERS:
        env = DSREnv(a.system, a.seed)
        if a.method == "beam":
            from dsr.planners import beam_plan
            res_by_pool = {}
            for p in pools_wanted:
                out = []
                for s in pools[p]:
                    r_ = beam_plan(env, s, cfg.get("width", 16), np.random.default_rng(a.seed))
                    out.append((*r_, beam_plan.last_ens))
                res_by_pool[p] = dict(srr=np.array([o[0] for o in out]), complete=np.array([float(o[1]) for o in out]), auc=np.array([o[2] for o in out]),
                                      violations=np.zeros(len(out)), steps=np.zeros(len(out)), attempts=np.zeros(len(out)), ret=np.zeros(len(out)), skips=np.zeros(len(out)),
                                      ens=np.array([o[3] for o in out]))
        else:
            agent = PlannerAgent(env, a.seed, a.method, **cfg)
            res_by_pool = {p: evaluate(agent, env, pools[p], time_decisions=a.time_decisions) for p in pools_wanted}
        result.update(train_time=0.0, episodes_run=0, train_log=[], n_params_actor=0, n_params_total=0)
    else:
        def make(env, seed):
            ag = LEARNED[a.method](env, seed, **cfg)
            if a.init_from:   # warm start (e.g. few-shot transfer from another system); size-dependent buffers are skipped
                ck = torch.load(os.path.join(ROOT, a.init_from) + ".pt", map_location="cpu", weights_only=False)
                st = ck["best"] if "best" in ck else ck
                st = {k: v for k, v in st.items() if not k.endswith(("src", "dst", "sw_f", "sw_t", "sw_line"))}
                (ag.net if hasattr(ag, "net") else ag.q).load_state_dict(st, strict=False)
                ag.pretrained = True if hasattr(ag, "pretrained") else None
            return ag
        runner = Runner(a.system, a.seed, make, pools["val_select"], a.episodes,
                        eval_every=a.eval_every, patience=a.patience, n_val=a.n_val, min_episodes=a.min_episodes, wall_limit_s=a.wall,
                        log_path=os.path.join(outdir, name + '.trainlog.json'))
        agent = runner.run(verbose=False)
        res_by_pool = {}
        for p in pools_wanted:
            res_by_pool[p] = evaluate(agent, runner.env, pools[p], time_decisions=a.time_decisions)
        # also evaluate the last (not best-val) checkpoint on the first pool for transparency
        best_state = agent.state(); agent.load(runner.final_state)
        res_final = evaluate(agent, runner.env, pools[pools_wanted[0]])
        agent.load(best_state)
        result.update(train_time=runner.train_time, episodes_run=runner.log[-1]["episode"] if runner.log else 0, train_log=runner.log,
                      n_params_actor=count_params(agent), n_params_total=total_params(agent), best_val_score=runner.best_val,
                      final_ckpt_eval={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res_final.items()})
        if a.lookahead:
            from agents.fgp_ext import Lookahead
            k, dpt = map(int, a.lookahead.split(","))
            la = Lookahead(agent, k, dpt)
            res_la = {p: evaluate(la, runner.env, pools[p], time_decisions=a.time_decisions) for p in pools_wanted}
            result["pools_lookahead"] = {p: {kk: v.tolist() for kk, v in r.items()} for p, r in res_la.items()}
            result["summary_lookahead"] = {p: summarize(r) for p, r in res_la.items()}
        if a.save_ckpt:
            torch.save(dict(best=best_state, final=runner.final_state), os.path.join(outdir, name + ".pt"))
    result["wall"] = time.time() - t0
    result["pools"] = {p: {k: v.tolist() for k, v in r.items()} for p, r in res_by_pool.items()}
    result["summary"] = {p: summarize(r) for p, r in res_by_pool.items()}
    json.dump(result, open(os.path.join(outdir, name + ".json"), "w"))
    print(name, json.dumps(result["summary"]), "wall %.0fs" % result["wall"])


if __name__ == "__main__":
    main()
