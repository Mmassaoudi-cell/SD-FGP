"""Generate the planner-expert dataset (TRAIN partition only): beam-search plans replayed into (state, feasible pairs, expert action,
reward-to-go) tuples used by the search-distilled candidate.  Usage: python scripts/make_expert.py ieee33 1500 24 16"""
import os, sys, pickle, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsr.env import DSREnv, LEVEL_SCALE
from dsr.planners import beam_plan
from agents.fgp import build_pairs

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def work(args):
    system, seed, width = args
    env = DSREnv(system, seed)
    rng = np.random.default_rng(50_000 + seed)
    scn = env.sample_scenario("train", rng)
    srr, comp, auc, plan = beam_plan(env, scn, width, np.random.default_rng(seed), return_plan=True)
    env.reset(scn)
    rows, rews = [], []
    prng = np.random.default_rng(0)
    for g, e in plan:
        if g < 0:                      # planned wait step (no feasible action)
            env.wait_step(); env.done = bool(env.t >= env.T or env.complete); continue
        d = build_pairs(env, prng)
        pos = np.where((d.e_idx == e) & (d.goal[:d.n] == g))[0]
        if len(pos) != 1:
            break
        lvl = env.commit(g, env.apply_edit(e)); env.done = bool(env.t >= env.T or env.complete)
        rows.append((d, int(pos[0]))); rews.append(LEVEL_SCALE * 19.0 * lvl)
    return dict(rows=rows, rewards=rews, srr=srr, auc=auc, complete=comp)


if __name__ == "__main__":
    system, n, width, nproc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    tag = sys.argv[5] if len(sys.argv) > 5 else ""
    t0 = time.time()
    with Pool(nproc) as p:
        out = p.map(work, [(system, s, width) for s in range(n)], chunksize=2)
    pickle.dump(out, open(os.path.join(ROOT, "data", f"expert{tag}_{system}.pkl"), "wb"))
    print(system, "scenarios", len(out), "steps", sum(len(o["rows"]) for o in out), "mean srr %.3f auc %.3f crr %.2f" % (
        np.mean([o["srr"] for o in out]), np.mean([o["auc"] for o in out]), np.mean([o["complete"] for o in out])), "%.0fs" % (time.time() - t0))
