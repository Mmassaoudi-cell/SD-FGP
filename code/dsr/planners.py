"""Model-based reference controllers (use the simulator): feasible-random, greedy, rollout (policy-improvement) planner."""
import numpy as np
from .env import DSREnv, MAX_RETRY


def _commit(env, goal, edit):
    env.commit(goal, env.apply_edit(edit))
    env.done = bool(env.t >= env.T or env.complete)


def greedy_choice(env, e, g, rng):
    n = env.net
    val = n.sigma[g] * n.P0[g] * env.dead0[g] + 1e-6 * rng.random(len(g))
    return int(np.argmax(val))


def run_episode(env, scn, choose, rng):
    env.reset(scn)
    while not env.done:
        e, g = env.feasible_set()
        if len(e) == 0:
            env.skip_step(); continue
        i = choose(env, e, g, rng)
        _commit(env, g[i], e[i])
    return env.srr, env.complete, env.auc()


def random_choice(env, e, g, rng):
    return int(rng.integers(len(e)))


def rollout_value(env, rng, base=greedy_choice, n_roll=1):
    """SRR reached by following the base policy to the end from the current state."""
    snap = env.snapshot()
    vals = []
    for _ in range(n_roll):
        while not env.done:
            e, g = env.feasible_set()
            if len(e) == 0:
                env.skip_step(); continue
            i = base(env, e, g, rng)
            _commit(env, g[i], e[i])
        vals.append(env.auc())
        env.restore(snap)
    return float(np.mean(vals))


def rollout_choice(max_cand=12, n_roll=1, base=greedy_choice):
    def choose(env, e, g, rng):
        # unique (goal, resulting switch vector) candidates, subsample for speed
        idx = np.arange(len(e))
        if len(idx) > max_cand:
            idx = rng.choice(idx, max_cand, replace=False)
        best, bi = -1.0, int(idx[0])
        snap = env.snapshot()
        for i in idx:
            _commit(env, g[i], e[i])
            v = rollout_value(env, rng, base, n_roll)
            env.restore(snap)
            if v > best:
                best, bi = v, int(i)
        return bi
    return choose


def beam_plan(env, scn, width=64, rng=None, return_plan=False, count_w=0.0):
    """Beam search over restoration sequences using the simulator (model-based upper reference; excluded from W/T/L).
    Dedupes on (energised set, switch vector); ranks by projected time-averaged service level (AUC).
    Returns (final SRR, complete, AUC[, plan]) of the best sequence; plan = [(goal, edit_index), ...]."""
    rng = rng or np.random.default_rng(0)
    env.reset(scn)
    frontier = [(env.snapshot(), [])]
    best = (env.auc(), env.restored_w, env.complete, [], env.snapshot())
    score = lambda auc, lvl: round(lvl, 6) + 1e-3 * auc   # final service level first, then time-averaged level
    depth = 0
    while frontier and depth < env.T:
        cand = []
        seen = set()
        for snap, plan in frontier:
            env.restore(snap)
            e, g = env.feasible_set()
            if len(e) == 0 and not env.complete:       # dead end: the only move is to wait (time-varying PV/load may unlock options)
                env.wait_step(); env.done = bool(env.t >= env.T or env.complete)
                cand.append((env.auc(), 0.0, env.snapshot(), env.restored_w, env.complete, plan + [(-1, -1)],
                             env.auc() + count_w * float((env.Y & env.dead0).sum()) / max(env.n_dead, 1)))
                continue
            for i in range(len(e)):
                env.restore(snap)
                _commit(env, g[i], e[i])
                key = (env.Y.tobytes(), env.k.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                cnt = float((env.Y & env.dead0).sum()) / max(env.n_dead, 1)
                cand.append((env.auc(), len(e) + rng.random() + 1e3 * count_w * cnt, env.snapshot(), env.restored_w, env.complete, plan + [(int(g[i]), int(e[i]))], env.auc() + count_w * cnt))
        if not cand:
            break
        cand.sort(key=lambda c: (-c[6], -c[1]))
        for c in cand:
            if score(c[0], c[3]) > score(best[0], best[1]) + 1e-12:
                best = (c[0], c[3], c[4], c[5], c[2])
        keep = cand[: width // 2] + [cand[i] for i in rng.choice(len(cand), min(len(cand), width - width // 2), replace=False)]
        frontier = [(c[2], c[5]) for c in keep]
        depth += 1
    env.restore(best[4])
    beam_plan.last_ens = env.ens()
    if return_plan:
        return best[1], bool(best[2]), best[0], best[3]
    return best[1], bool(best[2]), best[0]


def beam_first_action(env, width=8, horizon=5, rng=None):
    """Receding-horizon beam search from the CURRENT state.  Steps after the current one are evaluated with the noise-free forecast of
    demand and PV (non-anticipative).  Returns the index (goal, edit) of the first action of the best plan, or None."""
    rng = rng or np.random.default_rng(0)
    t0 = env.t
    snap0 = env.snapshot()
    env.fc_t0 = t0
    score = lambda auc, lvl: round(lvl, 6) + 1e-3 * auc
    frontier = [(snap0, None)]
    best, best_first = None, None
    depth = 0
    while frontier and depth < horizon and env.t < env.T:
        cand = []
        seen = set()
        for snap, first in frontier:
            env.restore(snap)
            if env.done:
                continue
            e, g = env.feasible_set()
            if len(e) == 0:
                continue
            for i in range(len(e)):
                env.restore(snap)
                _commit(env, g[i], e[i])
                key = (env.Y.tobytes(), env.k.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                f0 = first if first is not None else (int(g[i]), int(e[i]))
                cand.append((env.auc(), len(e) + rng.random(), env.snapshot(), env.restored_w, f0))
        if not cand:
            break
        cand.sort(key=lambda c: (-c[0], -c[1]))
        for c in cand:
            if best is None or score(c[0], c[3]) > best + 1e-12:
                best = score(c[0], c[3]); best_first = c[4]
        keep = cand[: width // 2] + [cand[i] for i in rng.choice(len(cand), min(len(cand), width - width // 2), replace=False)]
        frontier = [(c[2], c[4]) for c in keep]
        depth += 1
    env.fc_t0 = None
    env.restore(snap0)
    return best_first
