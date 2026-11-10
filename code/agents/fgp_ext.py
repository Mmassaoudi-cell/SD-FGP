"""Candidate extensions of the feasibility-constrained graph policy:
  FGPDistill  : planner-distilled initialisation (behaviour cloning of beam-search plans) + PPO fine-tuning with a BC regulariser
  FGPDQN      : off-policy n-step double-DQN over feasible pairs (pair-scorer used as Q-function)
  Lookahead   : inference-time value-guided look-ahead on top of a trained FGPPPO (policy prior + learned value)
"""
import copy, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fgp import FGPPPO, PairNet, build_pairs, collate, dist_from, PMAX
from dsr.env import LEVEL_SCALE

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def load_expert(system, max_scen=None, gamma=0.95, rs=0.1, files=None):
    """Expert tuples (PairData, expert_index, discounted expert return).  files: expert/dagger pickles under data/ (default: the v1 beam plans)."""
    files = files or [f"expert_{system}.pkl"]
    data = []
    for fn in files:
        raw = pickle.load(open(os.path.join(ROOT, "data", fn), "rb"))
        if len(raw) and isinstance(raw[0], tuple):          # DAgger rows are already (PairData, index, return)
            data += raw
            continue
        for o in raw[:max_scen]:
            rew = np.array(o["rewards"]) * rs
            ret = np.zeros(len(rew)); run = 0.0
            for j in range(len(rew) - 1, -1, -1):
                run = rew[j] + gamma * run; ret[j] = run
            for (d, idx), r in zip(o["rows"], ret):
                data.append((d, idx, float(r)))
    return data


class FGPDistill(FGPPPO):
    def __init__(self, env, seed, expert=None, bc_epochs=6, bc_w=0.3, bc_lr=1e-3, ppo=True, expert_frac=1.0, expert_files=None, **kw):
        super().__init__(env, seed, **kw)
        self.ppo = ppo
        self.expert = expert if expert is not None else load_expert(env.net.name, files=expert_files)
        if expert_frac < 1.0:
            self.expert = self.expert[: max(int(len(self.expert) * expert_frac), 8)]
        self.bc_epochs, self.bc_w, self.bc_lr = bc_epochs, bc_w, bc_lr
        self.pretrained = False

    def _bc_batch(self, idx):
        b = collate([self.expert[int(j)][0] for j in idx])
        tgt = torch.tensor([self.expert[int(j)][1] for j in idx])
        ret = torch.tensor([self.expert[int(j)][2] for j in idx], dtype=torch.float32)
        logits, value, gl = self.net(**b)
        logp = dist_from(logits, gl, b["goal"], b["mask"], self.hier)
        ce = -logp.gather(1, tgt.unsqueeze(1)).mean()
        return ce, F.mse_loss(value, ret)

    def pretrain(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.bc_lr)
        n = len(self.expert)
        for ep in range(self.bc_epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, 128):
                ce, vl = self._bc_batch(perm[s:s + 128])
                loss = ce + 0.5 * vl
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
        self.pretrained = True

    def train_episode(self, env, scn, ep):
        if not self.pretrained:
            self.pretrain()
        return super().train_episode(env, scn, ep)

    def _ppo_update(self):
        # standard PPO epoch(s) plus a BC regulariser on a random expert mini-batch
        if not self.ppo:
            return
        super()._ppo_update()
        for _ in range(4):
            ce, vl = self._bc_batch(np.random.randint(0, len(self.expert), 128))
            loss = self.bc_w * ce
            self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); self.opt.step()


class Lookahead:
    """Value-guided look-ahead around a trained FGPPPO.  depth d, width k: expand the policy's top-k pairs, score
    r + gamma * V(s') with the learned critic (leaf values from the network), choose the best first action."""

    def __init__(self, base, k=3, depth=2, gamma=0.95):
        self.base, self.k, self.depth, self.gamma = base, k, depth, gamma
        self.rng = np.random.default_rng(0)

    def begin_eval(self, env):
        pass

    def _value(self, env, d, depth):
        """Best discounted return achievable from the current state with `depth` more look-ahead levels."""
        if env.done:
            return 0.0
        if d is None:
            d = build_pairs(env, self.rng)
        if d.n == 0:
            return 0.0
        logp, value, _ = self.base._forward(d)
        if depth == 0:
            return float(value[0].item())
        top = torch.topk(logp[0], min(self.k, d.n)).indices.tolist()
        snap = env.snapshot()
        best = -1e9
        for i in top:
            lvl = env.commit(int(d.goal[i]), env.apply_edit(int(d.e_idx[i]))); env.done = bool(env.t >= env.T or env.complete)
            r = LEVEL_SCALE * 19.0 * lvl * self.base.rs
            v = r + self.gamma * self._value(env, None, depth - 1)
            env.restore(snap)
            best = max(best, v)
        return best

    def act_eval(self, env):
        d = build_pairs(env, self.rng)
        if d.n == 0:
            return None, None
        logp, value, _ = self.base._forward(d)
        top = torch.topk(logp[0], min(self.k, d.n)).indices.tolist()
        snap = env.snapshot()
        best, bi = -1e9, top[0]
        env.fc_t0 = env.t                      # non-anticipative: steps after the current one use the noise-free forecast
        try:
            for i in top:
                lvl = env.commit(int(d.goal[i]), env.apply_edit(int(d.e_idx[i]))); env.done = bool(env.t >= env.T or env.complete)
                r = LEVEL_SCALE * 19.0 * lvl * self.base.rs
                v = r + self.gamma * self._value(env, None, self.depth - 1)
                env.restore(snap)
                if v > best:
                    best, bi = v, i
        finally:
            env.fc_t0 = None
            env.restore(snap)
        return int(d.goal[bi]), env.apply_edit(int(d.e_idx[bi]))

    def state(self):
        return self.base.state()

    def load(self, st):
        self.base.load(st)


class FGPDQN:
    """n-step double DQN where Q(s, pair) is the PairNet score head (value head unused)."""

    def __init__(self, env, seed, encoder="gnn", H=64, layers=3, lr=5e-4, gamma=0.95, nstep=3, batch=64, buf=20000, warm_eps=20,
                 eps0=0.3, eps1=0.02, eps_decay_eps=1500, tau=0.01, reward_scale=0.1, update_every=1):
        self.env = env; self.rng = np.random.default_rng(seed)
        self.q = PairNet(env, H, layers, encoder, False); self.qt = copy.deepcopy(self.q)
        for p in self.qt.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.nstep, self.batch, self.cap, self.rs, self.tau = gamma, nstep, batch, buf, reward_scale, tau
        self.eps0, self.eps1, self.eps_decay, self.warm_eps = eps0, eps1, eps_decay_eps, warm_eps
        self.episodes = []           # list of episode dicts: data (PairData list), acts, rewards
        self.n_steps = 0; self.update_every = update_every; self.total_steps = 0

    def _scores(self, net, d, grad=False):
        b = collate([d])
        with torch.set_grad_enabled(grad):
            logits, _, _ = net(**b)
        return logits[0]

    def begin_eval(self, env):
        pass

    def act_eval(self, env):
        d = build_pairs(env, self.rng)
        if d.n == 0:
            return None, None
        i = int(torch.argmax(self._scores(self.q, d)).item())
        return int(d.goal[i]), env.apply_edit(int(d.e_idx[i]))

    def _eps(self, ep):
        return max(self.eps1, self.eps0 - (self.eps0 - self.eps1) * ep / self.eps_decay)

    def train_episode(self, env, scn, ep):
        env.reset(scn)
        data, acts, rews = [], [], []
        eps = self._eps(ep)
        while not env.done:
            d = build_pairs(env, self.rng)
            if d.n == 0:
                env.skip_step(); continue
            if self.rng.random() < eps or ep < self.warm_eps:
                i = int(self.rng.integers(d.n))
            else:
                i = int(torch.argmax(self._scores(self.q, d)).item())
            _, r, _, info = env.step(int(d.goal[i]), env.apply_edit(int(d.e_idx[i])), 19.0, 1.0)
            data.append(d); acts.append(i); rews.append(r * self.rs)
            self.total_steps += 1
            if ep >= self.warm_eps and self.n_steps > self.batch * 4 and self.total_steps % self.update_every == 0:
                self._update()
        self.episodes.append(dict(data=data, acts=acts, rews=rews))
        self.n_steps += len(data)
        while sum(len(e["data"]) for e in self.episodes) > self.cap and len(self.episodes) > 1:
            self.episodes.pop(0)
        return dict(ret=env.ret, srr=env.srr)

    def _update(self):
        B = self.batch
        lens = np.array([len(e["data"]) for e in self.episodes]); p = lens / lens.sum()
        eidx = self.rng.choice(len(self.episodes), B, p=p)
        cur, tgt_data, ys, acts, discs = [], [], [], [], []
        for ei in eidx:
            e = self.episodes[ei]; t = int(self.rng.integers(len(e["data"])))
            n = min(self.nstep, len(e["data"]) - t)
            G = sum((self.gamma ** j) * e["rews"][t + j] for j in range(n))
            cur.append(e["data"][t]); acts.append(e["acts"][t]); ys.append(G)
            if t + n < len(e["data"]):
                tgt_data.append(e["data"][t + n]); discs.append(self.gamma ** n)
            else:
                tgt_data.append(None); discs.append(0.0)
        b = collate(cur)
        q_all, _, _ = self.q(**b)
        q_sa = q_all.gather(1, torch.tensor(acts).unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            y = torch.tensor(ys, dtype=torch.float32)
            have = [i for i, d in enumerate(tgt_data) if d is not None]
            if have:
                bn = collate([tgt_data[i] for i in have])
                qo, _, _ = self.q(**bn); qt, _, _ = self.qt(**bn)
                a_star = qo.argmax(1, keepdim=True)
                nxt = qt.gather(1, a_star).squeeze(1)
                y[have] += torch.tensor([discs[i] for i in have], dtype=torch.float32) * nxt
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), 5.0); self.opt.step()
        with torch.no_grad():
            for p_, pt in zip(self.q.parameters(), self.qt.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p_)

    def state(self):
        return copy.deepcopy(self.q.state_dict())

    def load(self, st):
        self.q.load_state_dict(st)
        self.qt.load_state_dict(st)
