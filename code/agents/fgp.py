"""Feasibility-constrained graph policies (the new method family).

Pipeline per decision:
  1. exact physics feasibility screening of canonical (goal, <=2 switch flips) pairs  -> env.feasible_set()
  2. edge-conditioned GNN encoder over the (re)configurable network graph
  3. pointer/pair scoring head over the feasible pairs  (policy)  +  value head
Variants: flat pair policy ("flat"), goal-pointer then conditional edit ("hier"), MLP encoder ("mlp"), no feasibility mask
("nomask" handled in agents/nomask.py).
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import mlp

PMAX = 160


def _mask_fill(logits, mask):
    return logits.masked_fill(~mask, -1e9)


class GNNEncoder(nn.Module):
    def __init__(self, nf, ef, f_idx, t_idx, H=64, layers=3):
        super().__init__()
        self.H = H
        self.register_buffer("src", torch.cat([torch.as_tensor(f_idx), torch.as_tensor(t_idx)]).long())
        self.register_buffer("dst", torch.cat([torch.as_tensor(t_idx), torch.as_tensor(f_idx)]).long())
        self.L = len(f_idx)
        self.nin = nn.Linear(nf, H); self.ein = nn.Linear(ef, H)
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(3 * H, H), nn.ReLU(), nn.Linear(H, H)) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, H)) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(layers)])

    def forward(self, nodes, edges):
        h = F.relu(self.nin(nodes))              # (B,N,H)
        e = F.relu(self.ein(edges))              # (B,L,H)
        e2 = torch.cat([e, e], 1)                # directed copies
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            m = msg(torch.cat([h[:, self.src], h[:, self.dst], e2], -1))
            agg = torch.zeros_like(h).index_add_(1, self.dst, m)
            h = norm(h + upd(torch.cat([h, agg], -1)))
        return h, e


class PairNet(nn.Module):
    """Policy / value network over feasible (goal, edit) pairs."""

    def __init__(self, env, H=64, layers=3, encoder="gnn", hier=False):
        super().__init__()
        n = env.net
        self.H, self.hier, self.encoder_kind = H, hier, encoder
        self.S, self.N = env.S, env.N
        sw_line = torch.as_tensor(n.sw_line).long()
        self.register_buffer("sw_f", torch.as_tensor(n.f)[sw_line].long()); self.register_buffer("sw_t", torch.as_tensor(n.t)[sw_line].long())
        self.register_buffer("sw_line", sw_line)
        if encoder == "gnn":
            self.enc = GNNEncoder(env.node_feat_dim, env.edge_feat_dim, n.f, n.t, H, layers)
        else:   # flat MLP over concatenated features (benchmark/ablation)
            self.mlp_enc = mlp(env.N * env.node_feat_dim + env.L * env.edge_feat_dim, [256, 256], env.N * H)
            self.mlp_edge = mlp(env.N * env.node_feat_dim + env.L * env.edge_feat_dim, [256], env.L * H)
        self.sw_mlp = nn.Sequential(nn.Linear(3 * H, H), nn.ReLU(), nn.Linear(H, H))
        self.dirvec = nn.Parameter(torch.randn(2, H) * 0.1)
        self.noop = nn.Parameter(torch.randn(H) * 0.1)
        self.gnode = nn.Sequential(nn.Linear(H + 3, H), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(4 * H, 2 * H), nn.ReLU(), nn.Linear(2 * H, 1))
        self.goal_score = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, 1))
        self.value = nn.Sequential(nn.Linear(H + 3, H), nn.ReLU(), nn.Linear(H, 1))

    def encode(self, nodes, edges, glob):
        B = nodes.shape[0]
        if self.encoder_kind == "gnn":
            h, e = self.enc(nodes, edges)
        else:
            flat = torch.cat([nodes.reshape(B, -1), edges.reshape(B, -1)], -1)
            h = self.mlp_enc(flat).reshape(B, self.N, self.H); e = self.mlp_edge(flat).reshape(B, -1, self.H)
        pooled = h.mean(1)
        return h, e, pooled

    def pair_repr(self, h, e, pooled, glob, goal, s1, s2, d1, d2):
        B, P = goal.shape
        hg = torch.gather(h, 1, goal.unsqueeze(-1).expand(-1, -1, self.H))
        # switch embeddings (B,S,H)
        hf, ht = h[:, self.sw_f], h[:, self.sw_t]
        zs = self.sw_mlp(torch.cat([hf + ht, hf * ht, e[:, self.sw_line]], -1))
        def emb(s, d):
            sc = s.clamp(min=0)
            z = torch.gather(zs, 1, sc.unsqueeze(-1).expand(-1, -1, self.H)) + self.dirvec[(d > 0).long()]
            return z * (s >= 0).unsqueeze(-1)
        ed = emb(s1, d1) + emb(s2, d2)
        none = ((s1 < 0) & (s2 < 0)).unsqueeze(-1)
        ed = torch.where(none, self.noop.expand_as(ed), ed)
        return hg, ed

    def forward(self, nodes, edges, glob, goal, s1, s2, d1, d2, mask):
        h, e, pooled = self.encode(nodes, edges, glob)
        pg = torch.cat([pooled, glob], -1)
        value = self.value(pg).squeeze(-1)
        hg, ed = self.pair_repr(h, e, pooled, glob, goal, s1, s2, d1, d2)
        P = goal.shape[1]
        pooled_e = pooled.unsqueeze(1).expand(-1, P, -1)
        sc = self.score(torch.cat([hg, ed, pooled_e, hg * ed], -1)).squeeze(-1)
        logits = _mask_fill(sc, mask)
        if not self.hier:
            return logits, value, None
        # goal pointer: one logit per goal node, shared by all pairs with that goal (log-sum-exp over its edits)
        gl = self.goal_score(torch.cat([hg, pooled_e], -1)).squeeze(-1)
        return logits, value, gl


def dist_from(logits, gl, goal, mask, hier):
    """Log-probabilities over pairs.  hier: log pi(g) + log pi(e|g), pi(g) over goals having >=1 feasible pair."""
    if not hier:
        return F.log_softmax(logits, -1)
    # edit-conditional: softmax among pairs of the same goal
    B, P = goal.shape
    same = (goal.unsqueeze(2) == goal.unsqueeze(1)) & mask.unsqueeze(2) & mask.unsqueeze(1)    # (B,P,P)
    lg = logits.unsqueeze(1).expand(-1, P, -1).masked_fill(~same, -1e9)
    log_e_given_g = logits - torch.logsumexp(lg, -1)
    # goal distribution over first-occurrence pairs of each goal
    idx = torch.arange(P, device=goal.device)
    first = mask & ~((same & (idx.unsqueeze(0) < idx.unsqueeze(1)).unsqueeze(0)).any(-1))
    gl2 = gl.masked_fill(~first, -1e9)
    log_g = F.log_softmax(gl2, -1)          # on first occurrences
    # broadcast log pi(g) to all pairs with the same goal
    log_g_all = torch.gather(log_g.unsqueeze(1).expand(-1, P, -1), 2, torch.argmax(same.float() * first.unsqueeze(1).float(), -1, keepdim=True)).squeeze(-1)
    return (log_g_all + log_e_given_g).masked_fill(~mask, -1e9)


class PairData:
    """Container for one decision: arrays describing the state and feasible pairs."""
    __slots__ = ("nodes", "edges", "glob", "goal", "s1", "s2", "d1", "d2", "mask", "n", "e_idx")


def build_pairs(env, rng):
    e, g = env.feasible_set()
    d = PairData()
    d.nodes = env.node_features().astype(np.float32); d.edges = env.edge_features().astype(np.float32)
    d.glob = env.obs_flat()[-3:].astype(np.float32)
    n = len(e)
    d.n = n; d.e_idx = e
    sel = np.arange(n)
    if n > PMAX:      # deterministic truncation: enumeration order keeps all no-flip pairs, then single flips, then double flips
        e = e[:PMAX]; g = g[:PMAX]; d.e_idx = e; n = PMAX; d.n = n
    P = PMAX
    d.goal = np.zeros(P, np.int64); d.s1 = -np.ones(P, np.int64); d.s2 = -np.ones(P, np.int64)
    d.d1 = np.zeros(P, np.int64); d.d2 = np.zeros(P, np.int64); d.mask = np.zeros(P, bool)
    d.goal[:n] = g
    ed = env.edits[e] if n else np.zeros((0, 2), np.int64)
    d.s1[:n] = ed[:, 0]; d.s2[:n] = ed[:, 1]
    k = env.k
    d.d1[:n] = np.where(ed[:, 0] >= 0, np.where(k[np.maximum(ed[:, 0], 0)], -1, 1), 0)
    d.d2[:n] = np.where(ed[:, 1] >= 0, np.where(k[np.maximum(ed[:, 1], 0)], -1, 1), 0)
    d.mask[:n] = True
    return d


def collate(ds):
    T = lambda name, dt=None: torch.from_numpy(np.stack([getattr(x, name) for x in ds]))
    return dict(nodes=T("nodes"), edges=T("edges"), glob=T("glob"), goal=T("goal"), s1=T("s1"), s2=T("s2"), d1=T("d1"), d2=T("d2"), mask=T("mask"))


class FGPPPO:
    """PPO over feasible pairs.  variant: encoder in {gnn, mlp}, hier in {False, True}."""

    def __init__(self, env, seed, encoder="gnn", hier=False, H=64, layers=3, lr=3e-4, gamma=0.95, lam=0.95, clip=0.2, epochs=4,
                 mb=128, rollout_steps=1024, ent=0.01, vf=0.5, reward_scale=0.1, use_lookahead=False, max_grad=1.0):
        self.env = env; self.rng = np.random.default_rng(seed)
        self.net = PairNet(env, H, layers, encoder, hier)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip, self.epochs, self.mb = gamma, lam, clip, epochs, mb
        self.rollout_steps, self.ent, self.vf, self.rs, self.max_grad = rollout_steps, ent, vf, reward_scale, max_grad
        self.hier = hier
        self.buf = []
        self.total_steps = 0

    # ------------------------------------------------------------ acting
    def _forward(self, d, grad=False):
        b = collate([d])
        with torch.set_grad_enabled(grad):
            logits, value, gl = self.net(**b)
            logp = dist_from(logits, gl, b["goal"], b["mask"], self.hier)
        return logp, value, b

    def act_pair(self, d, deterministic):
        logp, value, b = self._forward(d)
        lp = logp[0]
        if deterministic:
            i = int(torch.argmax(lp).item())
        else:
            i = int(torch.multinomial(lp.exp().clamp_min(0), 1).item())
        return i, float(lp[i].item()), float(value[0].item())

    def begin_eval(self, env):
        self._tried = set(); self._last = None

    def act_eval(self, env):
        """Greedy choice among screened pairs; if the TRUE system rejected the previous attempt (model mismatch), exclude it and retry."""
        if env.fail_count == 0 or self._last is None:
            self._tried = set()
        else:
            self._tried.add(self._last)
        d = build_pairs(env, self.rng)
        if d.n == 0:
            return None, None
        logp, _, _ = self._forward(d)
        lp = logp[0].clone()
        for i in range(d.n):
            if (int(d.goal[i]), int(d.e_idx[i])) in self._tried:
                lp[i] = -1e9
        i = int(torch.argmax(lp).item())
        if lp[i] <= -1e8:
            return None, None
        self._last = (int(d.goal[i]), int(d.e_idx[i]))
        return int(d.goal[i]), env.apply_edit(int(d.e_idx[i]))

    # ------------------------------------------------------------ training
    def train_episode(self, env, scn, ep):
        env.reset(scn)
        ep_buf = []
        while not env.done:
            d = build_pairs(env, self.rng)
            if d.n == 0:
                env.skip_step(); continue
            i, lp, v = self.act_pair(d, False)
            r_before = env.restored_w
            goal, k = int(d.goal[i]), env.apply_edit(int(d.e_idx[i]))
            _, r_ex, _, info = env.step(goal, k, 19.0, 1.0)
            ep_buf.append((d, i, lp, v, r_ex * self.rs, float(info["terminated"])))
            self.total_steps += 1
        # GAE (episode complete; truncation bootstraps with 0 since horizon is part of the task)
        adv, last = [], 0.0
        for j in range(len(ep_buf) - 1, -1, -1):
            _, _, _, v, r, term = ep_buf[j]
            v_next = ep_buf[j + 1][3] if j + 1 < len(ep_buf) else 0.0
            delta = r + self.gamma * v_next - v
            last = delta + self.gamma * self.lam * last
            adv.append(last)
        adv = adv[::-1]
        for j, (d, i, lp, v, r, term) in enumerate(ep_buf):
            self.buf.append((d, i, lp, adv[j] + v, adv[j]))
        if len(self.buf) >= self.rollout_steps:
            self._ppo_update()
            self.buf = []
        return dict(ret=env.ret, srr=env.srr)

    def _ppo_update(self):
        n = len(self.buf)
        acts = torch.tensor([b[1] for b in self.buf]); old = torch.tensor([b[2] for b in self.buf], dtype=torch.float32)
        ret = torch.tensor([b[3] for b in self.buf], dtype=torch.float32); adv = torch.tensor([b[4] for b in self.buf], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.mb):
                idx = perm[s:s + self.mb]
                b = collate([self.buf[int(j)][0] for j in idx])
                logits, value, gl = self.net(**b)
                logp = dist_from(logits, gl, b["goal"], b["mask"], self.hier)
                lp = logp.gather(1, acts[idx].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(lp - old[idx])
                pl = -torch.min(ratio * adv[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                vl = F.mse_loss(value, ret[idx])
                p = logp.exp() * b["mask"]
                ent = -(p * logp.clamp_min(-30) * b["mask"]).sum(-1).mean()
                loss = pl + self.vf * vl - self.ent * ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.opt.step()

    def state(self):
        return copy.deepcopy(self.net.state_dict())

    def load(self, st):
        self.net.load_state_dict(st)
