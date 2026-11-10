"""Evaluate a saved checkpoint on scenario pools, optionally under model-mismatch perturbations / inference wrappers / zero-shot
transfer to another system.   python scripts/eval_ckpt.py --run results/repro/htd3pg_ieee33_s0 --pools test --out results/final_eval
TEST pools require DSR_ALLOW_TEST=1."""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsr.env import DSREnv
from agents.common import evaluate, summarize
from agents.registry import LEARNED
from scripts.run_one import load_pools, ROOT, count_params

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--pools", default="test"); ap.add_argument("--out", required=True)
ap.add_argument("--tag", default=""); ap.add_argument("--pert", default=""); ap.add_argument("--project", action="store_true")
ap.add_argument("--lookahead", default=""); ap.add_argument("--target_system", default=""); ap.add_argument("--time_decisions", action="store_true")
ap.add_argument("--ckpt_key", default="best"); ap.add_argument("--max_scen", type=int, default=0)
ap.add_argument("--rescreen", action="store_true"); ap.add_argument("--override", default="")
a = ap.parse_args()
if any(p.startswith("test") for p in a.pools.split(",")) and os.environ.get("DSR_ALLOW_TEST") != "1":
    sys.exit("TEST pools refused (DSR_ALLOW_TEST=1 required)")
run = a.run if os.path.isabs(a.run) else os.path.join(ROOT, a.run)
meta = json.load(open(run + ".json"))
system = a.target_system or meta["system"]
import dsr.env as E
if a.override:   # e.g. LOAD_REF=1.0,PV_REF=0.5,grid=0.7,T_EXTRA=5   (changes the simulator constants for this evaluation only)
    for kv in a.override.split(','):
        k_, v_ = kv.split('=')
        if k_ == 'LOAD_REF': E.LOAD_REF = float(v_)
        elif k_ == 'PV_REF': E.PV_REF = float(v_)
        elif k_ == 'T_EXTRA': E.T_EXTRA = int(v_)
env = DSREnv(system, meta["seed"])
if a.override and 'grid=' in a.override:
    env.net.grid_cap = float([kv.split('=')[1] for kv in a.override.split(',') if kv.startswith('grid=')][0]) * env.net.P0.sum()
env.meas_pv = a.rescreen
src_env = DSREnv(meta["system"], meta["seed"]) if a.target_system and a.target_system != meta["system"] else env
agent = LEARNED[meta["method"]](src_env, meta["seed"], **meta["cfg"])
ck = torch.load(run + ".pt", map_location="cpu", weights_only=False)
state = ck[a.ckpt_key] if isinstance(ck, dict) and "best" in ck and a.ckpt_key in ck else ck
if a.target_system and a.target_system != meta["system"]:      # zero-shot transfer of a graph policy (weights are size-independent)
    tgt = LEARNED[meta["method"]](env, meta["seed"], **meta["cfg"])
    net_src = agent.net if hasattr(agent, "net") else agent.q
    net_src.load_state_dict(state)
    sd = {k: v for k, v in net_src.state_dict().items() if not k.endswith(("src", "dst", "sw_f", "sw_t", "sw_line"))}
    (tgt.net if hasattr(tgt, "net") else tgt.q).load_state_dict(sd, strict=False)
    agent = tgt
else:
    agent.load(state)
if a.project and hasattr(agent, "project"):
    agent.project = True
if a.lookahead:
    from agents.fgp_ext import Lookahead
    k, d = map(int, a.lookahead.split(","))
    agent = Lookahead(agent, k, d)
if a.pert:
    p = dict(kv.split("=") for kv in a.pert.split(","))
    pert = {}
    for k, v in p.items():
        pert[k] = float(v)
    if "pvoff" in pert:   # disconnect the first n PV units
        n = int(pert.pop("pvoff")); vec = np.ones(len(env.net.pv_node)); vec[:n] = 0.0
        pert["pv"] = vec * pert.get("pv", 1.0)
    env.set_perturbation(pert)
pools = load_pools(system)
res = {}
for pn in a.pools.split(","):
    sc = pools[pn][: a.max_scen] if a.max_scen else pools[pn]
    res[pn] = evaluate(agent, env, sc, time_decisions=a.time_decisions)
out = dict(method=meta["method"], system=system, seed=meta["seed"], trained_on=meta["system"], tag=a.tag, pert=a.pert, project=a.project, lookahead=a.lookahead,
           pools={p: {k: v.tolist() for k, v in r.items()} for p, r in res.items()}, summary={p: summarize(r) for p, r in res.items()})
os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
name = f"{meta['method']}{a.tag}_{system}_s{meta['seed']}.json"
json.dump(out, open(os.path.join(ROOT, a.out, name), "w"))
print(name, json.dumps(out["summary"]))
