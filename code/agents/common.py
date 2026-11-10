"""Shared utilities: MLP, replay buffer, evaluation, generic training/early-stopping runner."""
import time, copy, json, os
import numpy as np
import torch
import torch.nn as nn
from dsr.env import DSREnv, MAX_RETRY

torch.set_num_threads(1)


def mlp(inp, hidden, out, act=nn.ReLU, last=None):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out))
    if last is not None:
        layers.append(last)
    return nn.Sequential(*layers)


def rec_from_obs(obs, N, nf=11):
    """Recoverable mask is node-feature column 1 of the flat observation (works on np or torch, batched or not)."""
    x = obs[..., : N * nf]
    x = x.reshape(*obs.shape[:-1], N, nf)
    return x[..., 1] > 0.5


class Replay:
    def __init__(self, cap, obs_dim, S):
        self.cap = cap; self.n = 0; self.i = 0
        self.o = np.zeros((cap, obs_dim), np.float32); self.o2 = np.zeros((cap, obs_dim), np.float32)
        self.g = np.zeros(cap, np.int64); self.a = np.zeros((cap, S), np.float32)
        self.u = np.zeros(cap, np.float32); self.v = np.zeros((cap, S), np.float32)  # continuous latent actions (feudal/td3)
        self.r = np.zeros(cap, np.float32); self.ri = np.zeros(cap, np.float32); self.d = np.zeros(cap, np.float32)

    def add(self, o, g, a, u, v, r, ri, o2, d):
        i = self.i
        self.o[i] = o; self.o2[i] = o2; self.g[i] = g; self.a[i] = a; self.u[i] = u; self.v[i] = v
        self.r[i] = r; self.ri[i] = ri; self.d[i] = d
        self.i = (i + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, B, rng):
        idx = rng.integers(0, self.n, B)
        t = lambda x: torch.from_numpy(x[idx])
        return t(self.o), t(self.g), t(self.a), t(self.u), t(self.v), t(self.r), t(self.ri), t(self.o2), t(self.d)


def eta_schedule(ep):
    """Paper Eqs. (47)-(48): eta_vio = min(0.5 + 0.0005*ep, 1), eta_r = 20 - eta_vio."""
    ev = min(0.5 + 0.0005 * ep, 1.0)
    return 20.0 - ev, ev


def evaluate(agent, env, scns, time_decisions=False):
    """Run the agent's deterministic/greedy policy on fixed scenarios.  Returns dict of per-scenario arrays."""
    srr, comp, steps, att, viol, ret, skips, lat, auc, ens = [], [], [], [], [], [], [], [], [], []
    for scn in scns:
        env.reset(scn)
        agent.begin_eval(env)
        while not env.done:
            t0 = time.perf_counter()
            goal, k = agent.act_eval(env)
            if time_decisions:
                lat.append(time.perf_counter() - t0)
            if goal is None:          # no feasible action exists: wait one step
                env.skip_step(); continue
            _, r, _, info = env.step(goal, k, *eta_schedule(10 ** 9))
        srr.append(env.srr); comp.append(float(env.complete)); steps.append(env.t); att.append(env.attempts)
        viol.append(env.n_viol); ret.append(env.ret); skips.append(env.skip_count); auc.append(env.auc()); ens.append(env.ens())
    out = dict(srr=np.array(srr), complete=np.array(comp), steps=np.array(steps), attempts=np.array(att),
               violations=np.array(viol), ret=np.array(ret), skips=np.array(skips), auc=np.array(auc), ens=np.array(ens))
    if time_decisions:
        out["latency_ms"] = np.array(lat) * 1e3
    return out


def summarize(res):
    return dict(srr_mean=float(res["srr"].mean() * 100), srr_min=float(res["srr"].min() * 100), srr_max=float(res["srr"].max() * 100),
                crr=float(res["complete"].mean() * 100), auc=float(res["auc"].mean() * 100), viol=float(res["violations"].mean()), skips=float(res["skips"].mean()),
                ret=float(res["ret"].mean()))


class Runner:
    """Generic training loop with periodic validation, best-checkpoint selection and patience-based early stopping."""

    def __init__(self, system, seed, make_agent, val_scns, episodes, eval_every=250, patience=None, n_val=60, log_path=None,
                 min_episodes=0, wall_limit_s=None):
        self.env = DSREnv(system, seed)
        self.system, self.seed = system, seed
        self.rng = np.random.default_rng(10_000 + seed)
        torch.manual_seed(seed); np.random.seed(seed)
        self.agent = make_agent(self.env, seed)
        self.val = val_scns[:n_val]
        self.episodes, self.eval_every, self.patience, self.min_episodes = episodes, eval_every, patience, min_episodes
        self.log_path, self.wall_limit = log_path, wall_limit_s
        self.log = []

    def run(self, verbose=True):
        best, best_state, bad = -1e9, None, 0
        t0 = time.time()
        train_time = 0.0
        ep_ret, ep_srr = [], []
        for ep in range(self.episodes):
            scn = self.env.sample_scenario("train", self.rng)
            t1 = time.time()
            info = self.agent.train_episode(self.env, scn, ep)
            train_time += time.time() - t1
            ep_ret.append(info["ret"]); ep_srr.append(info["srr"])
            if (ep + 1) % self.eval_every == 0 or ep + 1 == self.episodes:
                res = evaluate(self.agent, self.env, self.val)
                sm = summarize(res)
                score = sm["auc"] + 0.5 * sm["srr_mean"] + 0.2 * sm["crr"]
                rec = dict(episode=ep + 1, train_ret=float(np.mean(ep_ret[-self.eval_every:])), train_srr=float(np.mean(ep_srr[-self.eval_every:])),
                           wall=time.time() - t0, train_time=train_time, **sm, score=score)
                self.log.append(rec)
                if verbose:
                    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
                if self.log_path:
                    json.dump(self.log, open(self.log_path, "w"))
                if score > best + 1e-9:
                    best, bad = score, 0
                    best_state = self.agent.state()
                    if self.log_path:
                        try:
                            torch.save(best_state, self.log_path.replace('.trainlog.json', '.best.pt'))
                        except Exception:
                            pass
                else:
                    bad += 1
                if self.patience is not None and ep + 1 >= self.min_episodes and bad >= self.patience:
                    break
                if self.wall_limit and time.time() - t0 > self.wall_limit:
                    break
        self.final_state = self.agent.state()
        if best_state is not None:
            self.agent.load(best_state)
        self.train_time = train_time
        self.best_val = best
        return self.agent
