"""Distribution test systems for restoration: IEEE 33-bus (Baran-Wu, pandapower data) and a 123-node
balanced single-phase equivalent of the OpenDSS IEEE-123 feeder (assumption A1 in SOURCE_PAPER_AUDIT.md).
All quantities in per unit on S_base = 1 MVA (P, Q in MW / MVAr numerically)."""
import os, re
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


class Net:
    """Plain container of numpy arrays describing a switchable radial distribution system."""

    def __init__(self, name):
        self.name = name

    def finalize(self):
        N, L = self.N, self.L
        self.f = np.asarray(self.f, np.int64)
        self.t = np.asarray(self.t, np.int64)
        for a in ("r", "x", "smax", "P0", "Q0", "sigma", "pv_cap", "mt_cap"):
            setattr(self, a, np.asarray(getattr(self, a), np.float64))
        for a in ("zone", "pv_node", "mt_node", "sw_line", "sw_type", "sw_default"):
            setattr(self, a, np.asarray(getattr(self, a), np.int64))
        self.S = len(self.sw_line)
        self.line_sw = -np.ones(L, np.int64)
        for s, l in enumerate(self.sw_line):
            self.line_sw[l] = s
        # CSR adjacency over all lines
        deg = np.zeros(N + 1, np.int64)
        for l in range(L):
            deg[self.f[l] + 1] += 1
            deg[self.t[l] + 1] += 1
        self.adj_ptr = np.cumsum(deg)
        fill = self.adj_ptr[:-1].copy()
        self.adj_line = np.zeros(2 * L, np.int64)
        self.adj_nb = np.zeros(2 * L, np.int64)
        for l in range(L):
            for a, b in ((self.f[l], self.t[l]), (self.t[l], self.f[l])):
                self.adj_line[fill[a]] = l
                self.adj_nb[fill[a]] = b
                fill[a] += 1
        # default configuration after a fault: sectionalizing closed, ties open
        self.closed_default = np.ones(L, bool)
        for s, l in enumerate(self.sw_line):
            self.closed_default[l] = bool(self.sw_default[s])
        # lines that may fault: non-switch lines, assigned to the zone of the downstream (t) node
        self.fault_candidates = [
            [l for l in range(L) if self.line_sw[l] < 0 and self.zone[self.t[l]] == z and self.zone[self.f[l]] >= 0]
            for z in range(self.n_zones)
        ]
        self.slack = int(self.slack)
        self.total_P = float(self.P0.sum())
        return self


def _assign_sigma(N, seed, has_load):
    rng = np.random.default_rng(seed)
    sig = rng.choice([3.0, 2.0, 1.0], size=N)  # A2: seeded pseudo-random importance class
    sig[~has_load] = 1.0
    return sig


def build_33():
    import pandapower.networks as pn
    net = pn.case33bw()
    N = len(net.bus)
    zbase = 12.66 ** 2 / 1.0
    f, t, r, x = [], [], [], []
    for _, row in net.line.iterrows():
        f.append(int(row.from_bus)); t.append(int(row.to_bus))
        r.append(row.r_ohm_per_km * row.length_km / zbase)
        x.append(row.x_ohm_per_km * row.length_km / zbase)
    L = len(f)
    P0 = np.zeros(N); Q0 = np.zeros(N)
    for _, ld in net.load.iterrows():
        P0[int(ld.bus)] += ld.p_mw; Q0[int(ld.bus)] += ld.q_mvar
    key = {(a, b): i for i, (a, b) in enumerate(zip(f, t))}

    def line_of(a1, b1):  # 1-indexed paper buses
        a, b = a1 - 1, b1 - 1
        return key[(a, b)] if (a, b) in key else key[(b, a)]
    sect = [(3, 4), (8, 9), (13, 14), (29, 30)]
    ties = [(8, 21), (9, 15), (12, 22), (25, 29), (18, 33)]
    sw_line = [line_of(*p) for p in sect + ties]
    n = Net("ieee33")
    n.N, n.L, n.f, n.t, n.r, n.x = N, L, f, t, r, x
    n.smax = np.full(L, 4.0)
    n.P0, n.Q0 = P0, Q0
    n.sigma = _assign_sigma(N, 11, P0 > 0)
    zone = -np.ones(N, int)
    for z, nodes in enumerate([list(range(1, 6)) + [19, 20, 21, 22], list(range(23, 30)),
                               list(range(6, 14)), list(range(14, 19)) + [30, 31, 32, 33]]):
        for b in nodes:
            zone[b - 1] = z
    n.zone = zone; n.n_zones = 4
    n.sw_line = sw_line; n.sw_type = [0] * 4 + [1] * 5; n.sw_default = [1] * 4 + [0] * 5
    n.slack = 0
    n.pv_node = [b - 1 for b in (15, 22, 25, 33)]; n.pv_cap = [0.5, 0.2, 0.8, 0.3]
    n.mt_node = [b - 1 for b in (7, 24, 30, 32)]; n.mt_cap = [0.25] * 4
    n.vslack = 1.03
    n.vmin, n.vmax = 0.95, 1.05
    n.grid_cap = 0.8 * P0.sum()
    n.shunt_q = np.zeros(N)
    n.labels = [str(i + 1) for i in range(N)]
    return n.finalize()


# ------------------------------------------------------------------ IEEE 123 (OpenDSS) -> balanced equivalent
def _parse_matrix(txt):
    rows = re.findall(r"[\d\.eE\-\+\s]+", txt.replace("|", ";"))
    vals = [r.split() for r in txt.replace("[", "").replace("]", "").replace("(", "").replace(")", "").split("|")]
    return [[float(v) for v in row] for row in vals]


def _linecodes(path):
    codes = {}
    cur = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if line.startswith("!") or not line:
            continue
        m = re.match(r"new linecode\.(\w+)\s+nphases=(\d)", line, re.I)
        if m:
            cur = m.group(1); codes[cur] = {"ph": int(m.group(2))}
            continue
        if cur and line.startswith("~"):
            for key in ("rmatrix", "xmatrix"):
                m = re.search(key + r"\s*=\s*[\[\(](.*?)[\]\)]", line, re.I)
                if m:
                    codes[cur][key] = _parse_matrix(m.group(1))
    out = {}
    for k, c in codes.items():
        ph = c["ph"]
        def z(mat):
            diag = np.mean([mat[i][i] for i in range(ph)])
            if ph == 1:
                return diag
            off = np.mean([mat[i][j] if j <= i else mat[j][i] for i in range(ph) for j in range(ph) if i != j])
            return diag - off
        out[k] = (z(c["rmatrix"]), z(c["xmatrix"]), ph)
    return out


def build_123(seed=7):
    d = os.path.join(DATA_DIR, "ieee123")
    codes = _linecodes(os.path.join(d, "IEEELineCodes.DSS"))
    master = open(os.path.join(d, "IEEE123Master.dss"), encoding="utf-8", errors="ignore").read()
    bus = lambda s: re.sub(r"\..*$", "", s.strip().lower())
    raw = []  # (name, b1, b2, r_kft, x_kft, length_kft, phases)
    for m in re.finditer(r"^New Line\.(\w+)\s+(.*)$", master, re.I | re.M):
        name, rest = m.group(1), m.group(2)
        kv = dict(re.findall(r"(\w+)=([^\s]+)", rest))
        b1, b2 = bus(kv["Bus1"]), bus(kv["Bus2"])
        if "LineCode" in kv or "linecode" in {k.lower() for k in kv}:
            lc = kv.get("LineCode") or kv.get("linecode")
            rr, xx, ph = codes[lc]
            ln = float(kv["Length"])
            raw.append((name, b1, b2, rr * ln * 3.0 / ph, xx * ln * 3.0 / ph, ph))  # z_eq = z*3/phases (balanced equivalent)
        else:
            raw.append((name, b1, b2, 0.0, 0.0, 3))
    # contract zero-impedance (switches closed) and regulator pairs
    parent = {}
    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    for a, b in [("150", "150r"), ("150r", "149"), ("13", "152"), ("18", "135"), ("60", "160"), ("160", "160r"),
                 ("97", "197"), ("61", "61s"), ("9", "9r"), ("25", "25r")]:
        union(a, b)
    ties_open = [("151", "300"), ("54", "94")]  # original normally-open switches Sw7, Sw8
    lines = []
    for name, b1, b2, rr, xx, ph in raw:
        if name.lower().startswith("sw"):
            continue
        lines.append((find(b1), find(b2), rr, xx, ph, name))
    lines.append((find("61"), find("610"), 0.0635 * 1000 / 150 * 0.1, 0.0272 * 1000 / 150, 3, "XFM1"))
    # unique nodes
    names = sorted({a for l in lines for a in l[:2]}, key=lambda s: (len(s), s))
    # make sure the feeder head is node index 0
    head = find("150")
    names.remove(head); names.insert(0, head)
    idx = {nm: i for i, nm in enumerate(names)}
    N = len(names)
    # loads
    P0 = np.zeros(N); Q0 = np.zeros(N)
    for m in re.finditer(r"^New Load\.\w+\s+(.*)$", open(os.path.join(d, "IEEE123Loads.DSS"), encoding="utf-8", errors="ignore").read(), re.I | re.M):
        kv = {k.lower(): v for k, v in re.findall(r"(\w+)=([^\s]+)", m.group(1))}
        b = find(bus(kv["bus1"]))
        P0[idx[b]] += float(kv["kw"]) / 1000.0
        Q0[idx[b]] += float(kv["kvar"]) / 1000.0
    shunt = np.zeros(N)
    for b, kv in (("83", 600), ("88", 50), ("90", 50), ("92", 50)):
        shunt[idx[find(b)]] += kv / 1000.0
    f = [idx[l[0]] for l in lines]; t = [idx[l[1]] for l in lines]
    zb = 4.16 ** 2 / 1.0
    r = [l[2] / zb for l in lines]; x = [l[3] / zb for l in lines]
    smax = [{3: 3.0, 2: 1.5, 1: 0.8}[l[4]] for l in lines]
    # orient lines root->leaf with BFS from the head so that f is the upstream bus
    import collections
    nb = collections.defaultdict(list)
    for li, (a, b) in enumerate(zip(f, t)):
        nb[a].append(b); nb[b].append(a)
    depth = {0: 0}; q = collections.deque([0])
    while q:
        u = q.popleft()
        for v in nb[u]:
            if v not in depth:
                depth[v] = depth[u] + 1; q.append(v)
    assert len(depth) == N, "feeder is not connected"
    for li in range(len(f)):
        if depth[f[li]] > depth[t[li]]:
            f[li], t[li] = t[li], f[li]
    L0 = len(f)
    # coordinates for zones and tie placement
    coords = {}
    for line in open(os.path.join(d, "BusCoords.dat")):
        p = line.split()
        if len(p) >= 3:
            coords[find(p[0].lower())] = (float(p[1]), float(p[2]))
    xy = np.array([coords.get(nm, (np.nan, np.nan)) for nm in names])
    for i in range(N):  # nodes without coordinates inherit from the upstream neighbour
        if np.isnan(xy[i, 0]):
            for v in nb[i]:
                if not np.isnan(xy[v, 0]):
                    xy[i] = xy[v]; break
    # original sectionalizing lines: switches that were contracted (13-152, 18-135, 60-160, 97-197) become
    # explicit switchable lines by re-inserting them as near-zero impedance lines
    sect_pairs = [("13", "152"), ("18", "135"), ("60", "160"), ("97", "197")]
    # contracted pairs lose their line; rebuild by splitting: instead place sectionalizing switches on lines L10(8-13), L19(18-21), L61(60-62), L68(67-97)
    sect_names = ["L10", "L19", "L61", "L68", "L77"]
    sw_line = []
    name_to_line = {l[5]: i for i, l in enumerate(lines)}
    for nm in sect_names:
        sw_line.append(name_to_line[nm])
    # tie lines: the two original open switches + 7 new geometric ties between far-apart (in feeder) neighbours
    tie_pairs = [(idx[find(a)], idx[find(b)]) for a, b in ties_open]
    leaves = [i for i in range(1, N) if len(nb[i]) == 1]
    cand = []
    for a in range(N):
        for b in range(a + 1, N):
            if abs(depth[a] - depth[b]) < 0 or b in nb[a]:
                continue
            if np.isnan(xy[a, 0]) or np.isnan(xy[b, 0]):
                continue
            dist = float(np.hypot(*(xy[a] - xy[b])))
            cand.append((dist, a, b))
    cand.sort()
    # feeder (tree) distance via BFS per node would be O(N^2); compute with networkx
    import networkx as nx
    G = nx.Graph(); G.add_edges_from(zip(f, t))
    tree_dist = dict(nx.all_pairs_shortest_path_length(G))
    used = set(i for p in tie_pairs for i in p)
    for dist, a, b in cand:
        if len(tie_pairs) >= 9:
            break
        if a in used or b in used or tree_dist[a][b] < 12 or 0 in (a, b):
            continue
        tie_pairs.append((a, b)); used.update((a, b))
    assert len(tie_pairs) == 9, f"only {len(tie_pairs)} ties found"
    for a, b in tie_pairs:
        f.append(a); t.append(b)
        length_kft = max(0.3, float(np.hypot(*(xy[a] - xy[b]))) / 400.0)
        r.append(0.0867 * length_kft / zb); x.append(0.2042 * length_kft / zb)
        smax.append(3.0)
        sw_line.append(len(f) - 1)
    L = len(f)
    n = Net("ieee123")
    n.N, n.L, n.f, n.t, n.r, n.x, n.smax = N, L, f, t, r, x, smax
    n.P0, n.Q0 = P0, Q0
    n.shunt_q = shunt
    n.sigma = _assign_sigma(N, 13, P0 > 0)
    # zones: k-means on coordinates
    from scipy.cluster.vq import kmeans2
    cen, lab = kmeans2(np.nan_to_num(xy), 6, seed=seed, minit="++")
    n.zone = lab; n.n_zones = 6
    n.sw_line = sw_line; n.sw_type = [0] * 5 + [1] * 9; n.sw_default = [1] * 5 + [0] * 9
    n.slack = 0
    pvn = [idx[find(str(b))] for b in (11, 44, 56, 82, 91, 104)]
    n.pv_node = pvn; n.pv_cap = [0.2, 0.4, 0.4, 0.3, 0.4, 0.3]
    n.mt_node = [idx[find(str(b))] for b in (39, 46, 71, 75, 79, 96, 108)]; n.mt_cap = [0.25] * 7
    n.vslack = 1.04
    n.vmin, n.vmax = 0.95, 1.05
    n.grid_cap = 0.6 * P0.sum()   # calibrated on TRAIN scenarios (see DATA_AUDIT.md)
    n.labels = names
    n.xy = xy
    return n.finalize()


def build(name):
    return {"ieee33": build_33, "ieee123": build_123}[name]()


if __name__ == "__main__":
    for nm in ("ieee33", "ieee123"):
        n = build(nm)
        print(nm, "N", n.N, "L", n.L, "S", n.S, "P_total MW", round(n.total_P, 3), "zones", np.bincount(n.zone[n.zone >= 0]),
              "fault cands/zone", [len(c) for c in n.fault_candidates])
