"""Method registry: name -> factory(env, seed, **cfg).  Planner-type methods have no training stage."""
import numpy as np
from dsr.planners import greedy_choice, random_choice, rollout_choice, beam_plan, beam_first_action
from .fgp import FGPPPO
from .fgp_ext import FGPDistill, FGPDQN, Lookahead

LEARNED = {
    # ---- SD-FGP (proposed): screened pairs + GNN scorer + beam-search distillation + PPO fine-tuning
    "fgp_distill": lambda env, seed, **c: FGPDistill(env, seed, encoder="gnn", hier=False, **c),
    # ---- ablations / variants of the same family
    "fgp_flat": lambda env, seed, **c: FGPPPO(env, seed, encoder="gnn", hier=False, **c),          # PPO only (no distillation)
    "fgp_distill_mlp": lambda env, seed, **c: FGPDistill(env, seed, encoder="mlp", hier=False, **c),  # MLP instead of GNN encoder
    "ppo_mlp_masked": lambda env, seed, **c: FGPPPO(env, seed, encoder="mlp", **c),                 # screened MLP policy, PPO only
    "fgp_hier": lambda env, seed, **c: FGPPPO(env, seed, encoder="gnn", hier=True, **c),
    "fgp_dqn": lambda env, seed, **c: FGPDQN(env, seed, **c),
}

PLANNERS = {"greedy", "random_feasible", "rollout", "beam", "mpc"}


class PlannerAgent:
    """Wraps model-based references in the common evaluation API."""

    def __init__(self, env, seed, name, **cfg):
        self.env, self.name = env, name
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        if name == "mpc":
            self.choose = self._mpc
        else:
            self.choose = {"greedy": greedy_choice, "random_feasible": random_choice,
                           "rollout": rollout_choice(cfg.get("max_cand", 8), cfg.get("n_roll", 1))}[name]

    def _mpc(self, env, e, g, rng):
        first = beam_first_action(env, self.cfg.get("width", 8), self.cfg.get("horizon", 5), rng)
        if first is None:
            return 0
        m = np.where((g == first[0]) & (e == first[1]))[0]
        return int(m[0]) if len(m) else 0

    def begin_eval(self, env):
        pass

    def act_eval(self, env):
        e, g = env.feasible_set()
        if len(e) == 0:
            return None, None
        i = self.choose(env, e, g, self.rng)
        return int(g[i]), env.apply_edit(int(e[i]))

    def state(self):
        return None

    def load(self, st):
        pass
