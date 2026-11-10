"""Goal-conditioned distribution-service-restoration environment (reconstruction of the source paper's setting)."""
import os, hashlib, itertools
import numpy as np
from .network import build, DATA_DIR
from . import core

PROFILE_PATH = os.path.join(DATA_DIR, "profiles", "profiles.npz")
LOAD_REF = 0.9            # load_pu / LOAD_REF = demand multiplier (calibrated once on TRAIN scenarios, see DATA_AUDIT.md)
PV_REF = 0.6              # PV availability = rated * min(1, cf / PV_REF) (calibrated once on TRAIN scenarios)
NOISE_STD = 0.05          # 5 % Gaussian noise on load and PV (paper)
STEP_MIN = 3.0            # minutes per restoration step (paper)
K1, K2 = 5.0, 10.0        # hard-violation and goal-failure penalties (paper)
LEVEL_SCALE = 0.1         # per-step level-form reward scale (assumption A6')
MAX_RETRY = 10            # failed attempts before a forced skip step (assumption A5)
P_FAULT = 0.7             # probability that a zone has a fault (assumption A4)
T_EXTRA = 3              # horizon = number of outage nodes + T_EXTRA
MIN_NOON_CF = 0.15        # drop storm days where black-start PV cannot even energise its own bus


class Profiles:
    def __init__(self):
        z = np.load(PROFILE_PATH, allow_pickle=True)
        self.dates = z["dates"]; self.pv_cf = z["pv_cf"]; self.load_pu = z["load_pu"]; self.split = z["split"]
        ok = self.pv_cf[:, 23] >= MIN_NOON_CF
        self.pool = {s: np.where((self.split == k) & ok)[0] for s, k in (("train", 0), ("val", 1), ("test", 2))}

    def cf(self, day, hour):
        """PV capacity factor at local hour (float) via linear interpolation of 30-min data (index i = interval ending (i+1)/2 h)."""
        x = hour * 2.0 - 1.0
        i0 = int(np.floor(x)); w = x - i0
        a = self.pv_cf[day, max(0, min(47, i0))]; b = self.pv_cf[day, max(0, min(47, i0 + 1))]
        return float((1 - w) * a + w * b)

    def load(self, day, hour):
        x = hour; i0 = int(np.floor(x)); w = x - i0
        a = self.load_pu[day, min(23, i0)]; b = self.load_pu[day, min(23, i0 + 1)]
        return float((1 - w) * a + w * b)


_PROF = None
def profiles():
    global _PROF
    if _PROF is None:
        _PROF = Profiles()
    return _PROF


def fault_split(faults):
    """Deterministic split of a fault set: 70 % train, 15 % val, 15 % test (hash of the sorted fault tuple)."""
    h = int(hashlib.md5(str(tuple(sorted(faults))).encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


class Scenario:
    def __init__(self, faults, day, hour, seed, split):
        self.faults = tuple(sorted(faults)); self.day = int(day); self.hour = float(hour); self.seed = int(seed); self.split = split


class DSREnv:
    def __init__(self, system="ieee33", seed=0, net=None):
        self.net = net if net is not None else build(system)
        self.prof = profiles()
        n = self.net
        self.N, self.L, self.S = n.N, n.L, n.S
        self.rng = np.random.default_rng(seed)
        self._V = np.zeros(self.N); self._Sb = np.zeros(self.L)
        self._rec = np.zeros(self.N, np.bool_)
        self.is_pv = np.zeros(self.N); self.is_pv[n.pv_node] = 1
        self.is_mt = np.zeros(self.N); self.is_mt[n.mt_node] = 1
        self.pv_pu = np.zeros(self.N); self.mt_pu = np.zeros(self.N)
        self.mt_pu[n.mt_node] = n.mt_cap
        self.node_feat_dim = 11
        self.edge_feat_dim = 8
        self.scn = None
        self.pert = None   # model-mismatch perturbation of the TRUE system (robustness experiments)
        self.fc_t0 = None      # when set, steps after fc_t0 use the noise-free forecast (non-anticipative planning)
        self.meas_pv = False   # screening uses the measured (true) PV availability instead of the nominal forecast
        # enumerated switch edits (<= 2 flips); index 0 = no change
        S = self.S
        edits = [(-1, -1)] + [(s, -1) for s in range(S)] + list(itertools.combinations(range(S), 2))
        self.edits = np.array(edits, np.int64)
        self.edit_mask_cache = None

    # ------------------------------------------------------------------ scenarios
    def sample_scenario(self, split, rng, hour=12.0, zones=None, max_faults=1, require_zone=None):
        """zones: restrict faults to these zones (generalisation test); max_faults: faults per zone (1 = training protocol,
        2 = severe-event pool); require_zone: at least one fault must lie in this zone."""
        n = self.net
        excl = os.environ.get("DSR_EXCLUDE_ZONE")          # leave-zone-out experiment: training faults never occur in this zone
        if split == "train" and zones is None and excl not in (None, ""):
            zones = [z for z in range(n.n_zones) if z != int(excl)]
        while True:
            faults = []
            for z in range(n.n_zones):
                if zones is not None and z not in zones:
                    continue
                if rng.random() < P_FAULT and n.fault_candidates[z]:
                    k = 1 if max_faults == 1 else int(rng.integers(1, max_faults + 1))
                    k = min(k, len(n.fault_candidates[z]))
                    faults += [int(x) for x in rng.choice(n.fault_candidates[z], size=k, replace=False)]
            if require_zone is not None and not any(n.zone[n.t[l]] == require_zone for l in faults):
                continue
            if not faults or fault_split(faults) != split:
                continue
            day = int(rng.choice(self.prof.pool[split]))
            scn = Scenario(faults, day, hour, int(rng.integers(1 << 30)), split)
            if self._valid(scn):
                return scn

    def _valid(self, scn):
        self._setup(scn)
        return self.valid

    def _setup(self, scn):
        n = self.net
        self.scn = scn
        self.exist = np.ones(self.L, np.bool_)
        for l in scn.faults:
            self.exist[l] = False
        rng = np.random.default_rng(scn.seed)
        self.Tmax = self.N + 4
        self._pv_cap_nodes = n.pv_cap
        T = self.Tmax + 2
        hours = scn.hour + np.arange(T) * STEP_MIN / 60.0
        cf = np.array([self.prof.cf(scn.day, h) for h in hours])
        ld = np.array([self.prof.load(scn.day, h) for h in hours]) / LOAD_REF
        self.cf_t = np.clip(cf[:, None] * (1 + NOISE_STD * rng.standard_normal((T, len(n.pv_node)))), 0, 1.0)
        self.m_t = ld[:, None] * np.clip(1 + NOISE_STD * rng.standard_normal((T, self.N)), 0.7, 1.3)
        self.m_fc = ld[:, None] * np.ones((T, self.N))
        self.cf_fc = np.clip(cf[:, None] * np.ones((T, len(n.pv_node))), 0, 1.0)
        # initial energised set
        closed0 = n.closed_default & self.exist
        Y = np.zeros(self.N, np.bool_)
        stack = [n.slack]; Y[n.slack] = True
        while stack:
            u = stack.pop()
            for k in range(n.adj_ptr[u], n.adj_ptr[u + 1]):
                l = n.adj_line[k]; v = n.adj_nb[k]
                if closed0[l] and not Y[v]:
                    Y[v] = True; stack.append(v)
        pv0 = n.pv_cap * np.minimum(1.0, self.cf_t[0] / PV_REF)
        for j, p in enumerate(n.pv_node):   # a black-start PV bus energises itself only if its output covers its own demand
            if pv0[j] >= 1.05 * n.P0[p] * self.m_t[0, p]:
                Y[p] = True
        self.Y0 = Y.copy()
        self.dead0 = ~Y
        self.n_dead = int(self.dead0.sum())
        # all outage nodes reachable through non-faulted lines?
        reach = Y.copy(); stack = list(np.where(Y)[0])
        while stack:
            u = stack.pop()
            for k in range(n.adj_ptr[u], n.adj_ptr[u + 1]):
                l = n.adj_line[k]; v = n.adj_nb[k]
                if self.exist[l] and not reach[v]:
                    reach[v] = True; stack.append(v)
        self.restorable0 = reach & self.dead0
        self.W = float((n.sigma * n.P0)[self.dead0].sum())
        self.valid = bool(self.dead0.any() and reach.all() and self.W > 0 and self.n_dead >= 2)
        if self.valid:
            self.t = 0
            self.Y = Y.copy()
            self.k = n.sw_default.astype(np.bool_).copy()
            pert, self.pert = self.pert, None      # scenario validity is defined on the nominal system
            code = self._eval(self.Y, self.closed())[0]
            self.pert = pert
            if code != 0:
                self.valid = False

    def reset(self, scn):
        self._setup(scn)
        assert self.valid, "invalid scenario"
        n = self.net
        self.t = 0
        self.T = min(self.n_dead + T_EXTRA, self.Tmax)
        self.Y = self.Y0.copy()
        self.k = n.sw_default.astype(np.bool_).copy()
        self.attempts = 0
        self.fail_count = 0
        self.skip_count = 0
        self.ret = 0.0
        self.restored_w = 0.0
        self.auc_sum = 0.0
        self.ens_sum = 0.0
        self.fc_t0 = None
        self.done = False
        self.n_viol = 0
        self.eta_vio = 1.0
        self._refresh()
        return self.obs_flat()

    # ------------------------------------------------------------------ physics helpers
    def closed(self, k=None):
        n = self.net
        k = self.k if k is None else k
        c = self.exist.copy()
        for s in range(self.S):
            if not k[s]:
                c[n.sw_line[s]] = False
        return c

    def demand(self, t=None):
        t = self.t if t is None else t
        n = self.net
        m = self.m_t[min(t, len(self.m_t) - 1)]
        if self.fc_t0 is not None and t > self.fc_t0:
            m = self.m_fc[min(t, len(self.m_fc) - 1)]
        P = n.P0 * m
        Q = n.Q0 * m - n.shunt_q
        return P, Q

    def pv_avail(self, t=None):
        t = self.t if t is None else t
        cft = self.cf_t[min(t, len(self.cf_t) - 1)]
        if self.fc_t0 is not None and t > self.fc_t0:
            cft = self.cf_fc[min(t, len(self.cf_fc) - 1)]
        return self.net.pv_cap * np.minimum(1.0, cft / PV_REF)

    def set_perturbation(self, pert):
        """pert = dict(load=float, pv=array(len(pv)) or float, z=float) applied to the TRUE system only; the controller's
        feasibility model (feasible_set) keeps the nominal parameters."""
        self.pert = pert

    def _eval(self, Y, closed, t=None):
        n = self.net
        P, Q = self.demand(t)
        pv = self.pv_avail(t); r, x = n.r, n.x
        if self.pert:
            P = P * self.pert.get("load", 1.0); Q = Q * self.pert.get("load", 1.0)
            pv = pv * self.pert.get("pv", 1.0)
            r = r * self.pert.get("z", 1.0); x = x * self.pert.get("z", 1.0)
        return core.eval_state(n.N, n.L, n.f, n.t, r, x, n.smax, n.adj_ptr, n.adj_line, n.adj_nb, closed, Y, P, Q,
                               n.slack, n.grid_cap, n.vslack, n.pv_node, pv, n.mt_node, n.mt_cap,
                               n.vmin, n.vmax, self._V, self._Sb)

    def _refresh(self):
        """Recompute state-dependent quantities (voltages, recoverable set) for observation / masks."""
        n = self.net
        self._eval(self.Y, self.closed())
        core.recoverable_goals(n.N, n.L, n.f, n.t, self.exist, self.Y, self._rec)
        self.srr = float((n.sigma * n.P0)[self.Y & self.dead0].sum() / self.W)
        self.complete = bool(self.srr >= 1.0 - 1e-9)   # CRR definition of the paper: weighted service restoration rate = 100 %

    def recoverable(self):
        return self._rec.copy()

    # ------------------------------------------------------------------ transitions
    def try_action(self, goal, k_new):
        """Pure evaluation: returns (code, vviol, sviol)."""
        Y2 = self.Y.copy(); Y2[goal] = True
        code, vv, sv, _ = self._eval(Y2, self.closed(np.asarray(k_new, np.bool_)))
        return code, vv, sv

    def commit(self, goal, k_new):
        """Apply a verified-feasible action: energise the goal, set switches, advance one 3-minute step.  Returns the level after the step."""
        n = self.net
        self.Y[goal] = True
        self.k = np.asarray(k_new, np.bool_).copy()
        self.t += 1
        self.fail_count = 0
        if self.dead0[goal]:
            self.restored_w += float(n.sigma[goal] * n.P0[goal] / self.W)
        self.auc_sum += self.restored_w
        self._acc_ens()
        self._refresh()
        return self.restored_w

    def wait_step(self):
        """Time advances with no state change (forced skip / no feasible action)."""
        self.t += 1; self.fail_count = 0; self.skip_count += 1
        self.auc_sum += self.restored_w
        self._acc_ens()
        self._refresh()

    def _unserved_kwh_per_step(self):
        P, _ = self.demand()
        return float(P[self.dead0 & ~self.Y].sum()) * (STEP_MIN / 60.0) * 1000.0

    def _acc_ens(self):
        self.ens_sum += self._unserved_kwh_per_step()

    def ens(self):
        """Energy not served (kWh) over the episode horizon (unserved demand of the outage nodes integrated over the 3-minute steps)."""
        rem = max(self.T - self.t, 0)
        return float(self.ens_sum + rem * self._unserved_kwh_per_step())

    def auc(self):
        """Time-averaged weighted service level over the episode horizon (objective of paper Eq. (1), normalised)."""
        rem = max(self.T - self.t, 0)
        return float((self.auc_sum + rem * self.restored_w) / max(self.T, 1))

    def step(self, goal, k_new, eta_r=19.0, eta_vio=1.0):
        """One attempt.  Returns obs, r_ex, r_in, info.  An infeasible attempt does not advance the system state (paper) but
        consumes one retry; after MAX_RETRY the step is skipped (time advances)."""
        n = self.net
        assert not self.done
        goal = int(goal)
        k_new = np.asarray(k_new, np.bool_)
        self.attempts += 1
        ok_goal = bool(self._rec[goal])
        if ok_goal:
            code, vv, sv = self.try_action(goal, k_new)
        else:
            code, vv, sv = 2, 0.0, 0.0
        info = {"code": code, "success": code == 0}
        if code == 0:
            level = self.commit(goal, k_new)
            r_ex = LEVEL_SCALE * eta_r * level          # Eq. (18), level form, scaled (A6')
            r_in = 0.0
        else:
            self.n_viol += 1
            self.fail_count += 1
            r_ex = -eta_vio * (vv + sv + K1)
            r_in = -K2
            if self.fail_count >= MAX_RETRY:  # forced skip: time advances, state unchanged
                self.wait_step()
                r_ex += LEVEL_SCALE * eta_r * self.restored_w
                info["skipped"] = True
        self.ret += r_ex
        trunc = self.t >= self.T
        term = self.complete
        self.done = bool(trunc or term)
        info.update(terminated=bool(term), truncated=bool(trunc and not term), srr=self.srr, complete=self.complete)
        return self.obs_flat(), r_ex, r_in, info

    # ------------------------------------------------------------------ observations
    def node_features(self):
        n = self.net
        P, Q = self.demand()
        pv = np.zeros(self.N); pv[n.pv_node] = self.pv_avail()
        F = np.stack([
            self.Y.astype(np.float64), self._rec.astype(np.float64), P / 0.2, Q / 0.2, n.sigma / 3.0, self._V,
            self.is_pv, pv / 0.5, self.is_mt, self.dead0.astype(np.float64), (np.arange(self.N) == n.slack).astype(np.float64)
        ], axis=1)
        return F

    def edge_features(self):
        n = self.net
        cl = self.closed()
        is_sw = (n.line_sw >= 0).astype(np.float64)
        is_tie = np.zeros(self.L); is_tie[n.sw_line[n.sw_type == 1]] = 1
        en = (cl & self.Y[n.f] & self.Y[n.t]).astype(np.float64)
        return np.stack([self.exist.astype(np.float64), cl.astype(np.float64), is_sw, is_tie, n.r * 20, n.x * 20,
                         self._Sb / np.maximum(n.smax, 1e-9), en], axis=1)

    def obs_flat(self):
        F = self.node_features()
        g = np.array([self.t / max(self.T, 1), (self.n_dead - (self.Y & self.dead0).sum()) / max(self.n_dead, 1), self.fail_count / MAX_RETRY])
        return np.concatenate([F.ravel(), self.edge_features()[:, :2].ravel(), self.k.astype(np.float64), g]).astype(np.float32)

    @property
    def obs_dim(self):
        return self.N * self.node_feat_dim + 2 * self.L + self.S + 3

    # ------------------------------------------------------------------ structured action set (proposed method)
    def _screen_pv(self):
        pv = self.pv_avail()
        if self.meas_pv and self.pert:
            pv = pv * self.pert.get('pv', 1.0)
        return pv

    def feasible_set(self):
        """All (edit_index, goal) pairs that are feasible from the current state."""
        n = self.net
        P, Q = self.demand()
        out_e = np.empty(len(self.edits) * 12, np.int64); out_g = np.empty_like(out_e); out_k = np.empty_like(out_e)
        closed0 = self.closed()
        m = core.feasible_pairs(n.N, n.L, n.f, n.t, n.r, n.x, n.smax, n.adj_ptr, n.adj_line, n.adj_nb, closed0, self.Y, P, Q,
                                n.slack, n.grid_cap, n.vslack, n.pv_node, self._screen_pv(), n.mt_node, n.mt_cap, n.vmin, n.vmax,
                                n.sw_line, self.edits, self.exist, out_e, out_g, out_k)
        return out_e[:m].copy(), out_g[:m].copy()

    def snapshot(self):
        return (self.Y.copy(), self.k.copy(), self.t, self.fail_count, self.attempts, self.skip_count, self.ret,
                self.restored_w, self.done, self.n_viol, self.auc_sum, self.ens_sum)

    def restore(self, snap):
        (Y, k, self.t, self.fail_count, self.attempts, self.skip_count, self.ret, self.restored_w, self.done, self.n_viol, self.auc_sum, self.ens_sum) = snap
        self.Y = Y.copy(); self.k = k.copy()
        self._refresh()

    def skip_step(self):
        """No feasible action: the agent keeps trying until the retry cap forces a skip (time advances)."""
        self.wait_step()   # a screened controller knows nothing is feasible and simply waits (no infeasible attempt is made)
        self.ret += LEVEL_SCALE * 19.0 * self.restored_w
        self.done = bool(self.t >= self.T or self.complete)

    def apply_edit(self, e):
        k = self.k.copy()
        for s in self.edits[e]:
            if s >= 0:
                k[s] = not k[s]
        return k


def make_pool(system, split, n, seed):
    env = DSREnv(system)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        scn = env.sample_scenario(split, rng)
        if not any(scn.faults == o.faults and scn.day == o.day for o in out):
            out.append(scn)
    return out
