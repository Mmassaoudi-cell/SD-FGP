"""Numba kernels: feasibility / power-flow evaluation of an energised set under a switch configuration.

State convention (assumption A5 in the audit): a node is *energised* iff its load is restored (y_i = 1, source buses
are energised at t = 0).  The energised sub-graph = energised nodes + closed lines between them.
Failure codes:  0 ok, 1 loop (radiality), 2 energised island without a grid-forming source (monotone-restoration /
connectivity), 3 supply capacity exceeded, 4 voltage limit, 5 branch limit.
"""
import numpy as np
from numba import njit

OK, LOOP, NOREF, CAP, VOLT, BRANCH = 0, 1, 2, 3, 4, 5


@njit(cache=True)
def _find(par, a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a


@njit(cache=True)
def eval_state(N, L, f, t, r, x, smax, adj_ptr, adj_line, adj_nb, closed, Y, P, Q, slack, grid_cap, vslack,
               pv_node, pv_avail, mt_node, mt_cap, vmin, vmax, V_out, S_out):
    """Evaluate one configuration.  P, Q: per-node demand (already scaled) of energised nodes (+ shunt).
    Returns (code, v_violation, s_violation, supply_slack).  V_out/S_out are filled with node voltages / branch
    apparent power (0 for inactive)."""
    for i in range(N):
        V_out[i] = 0.0
    for l in range(L):
        S_out[l] = 0.0
    # ---- loop detection among closed energised lines
    par = np.arange(N)
    for l in range(L):
        if closed[l] and Y[f[l]] and Y[t[l]]:
            a = _find(par, f[l]); b = _find(par, t[l])
            if a == b:
                return 1, 0.0, 0.0, 0.0
            par[a] = b
    # ---- roots: slack first, then PV buses by capacity (descending)
    nroot = 0
    roots = np.empty(len(pv_node) + 1, np.int64)
    if Y[slack]:
        roots[nroot] = slack; nroot += 1
    pv_order = np.argsort(-pv_avail)
    for k in range(len(pv_node)):
        j = pv_order[k]
        if Y[pv_node[j]] and pv_node[j] != slack:
            roots[nroot] = pv_node[j]; nroot += 1
    comp = -np.ones(N, np.int64)
    order = np.empty(N, np.int64)
    parent = -np.ones(N, np.int64)
    pline = -np.ones(N, np.int64)
    cstart = np.empty(nroot + 1, np.int64)
    n_ord = 0
    ncomp = 0
    for ri in range(nroot):
        rt = roots[ri]
        if comp[rt] >= 0:
            continue
        cstart[ncomp] = n_ord
        comp[rt] = ncomp
        order[n_ord] = rt; n_ord += 1
        head = cstart[ncomp]
        while head < n_ord:
            u = order[head]; head += 1
            for k in range(adj_ptr[u], adj_ptr[u + 1]):
                l = adj_line[k]; v = adj_nb[k]
                if closed[l] and Y[v] and comp[v] < 0:
                    comp[v] = ncomp; parent[v] = u; pline[v] = l
                    order[n_ord] = v; n_ord += 1
        ncomp += 1
    cstart[ncomp] = n_ord
    for i in range(N):
        if Y[i] and comp[i] < 0:
            return 2, 0.0, 0.0, 0.0  # energised node without a grid-forming source
    # ---- sources per node
    gen = np.zeros(N)
    vviol = 0.0
    sviol = 0.0
    supply_slack = 1e9
    code = 0
    Vc = np.zeros(N, np.complex128)
    Ib = np.zeros(N, np.complex128)  # branch current into node i from its parent
    for c in range(ncomp):
        s0 = cstart[c]; s1 = cstart[c + 1]
        rt = order[s0]
        grid = (rt == slack)
        load = 0.0
        for k in range(s0, s1):
            load += P[order[k]]
        # DER capacities in component (PV available + MT capacity; MTs only matter inside energised comps)
        cap_pv = 0.0
        for j in range(len(pv_node)):
            if comp[pv_node[j]] == c:
                cap_pv += pv_avail[j]
        cap_mt = 0.0
        for j in range(len(mt_node)):
            if comp[mt_node[j]] == c:
                cap_mt += mt_cap[j]
        cap_der = cap_pv + cap_mt
        loss = 0.0
        for outer in range(3):
            need = load + loss
            phi = 1.0
            if cap_der > 1e-12:
                phi = need / cap_der
                if phi > 1.0:
                    phi = 1.0
            # root node: for grid comps the grid is the reference; for islands the root PV is
            for k in range(s0, s1):
                gen[order[k]] = 0.0
            for j in range(len(pv_node)):
                if comp[pv_node[j]] == c and pv_node[j] != rt:
                    gen[pv_node[j]] += phi * pv_avail[j]
            for j in range(len(mt_node)):
                if comp[mt_node[j]] == c and mt_node[j] != rt:
                    gen[mt_node[j]] += phi * mt_cap[j]
            vroot = vslack if grid else 1.0
            for k in range(s0, s1):
                Vc[order[k]] = vroot
            for it in range(15):
                for k in range(s0, s1):
                    Ib[order[k]] = 0.0
                for k in range(s1 - 1, s0, -1):
                    u = order[k]
                    sload = complex(P[u] - gen[u], Q[u])
                    Ii = np.conj(sload / Vc[u])
                    Itot = Ib[u] + Ii
                    Ib[u] = Itot
                    Ib[parent[u]] += Itot
                    # note: Ib[u] now holds the branch current into u from its parent
                dmax = 0.0
                for k in range(s0 + 1, s1):
                    u = order[k]
                    l = pline[u]
                    vn = Vc[parent[u]] - complex(r[l], x[l]) * Ib[u]
                    dd = abs(vn - Vc[u])
                    if dd > dmax:
                        dmax = dd
                    Vc[u] = vn
                if dmax < 1e-8:
                    break
            loss = 0.0
            for k in range(s0 + 1, s1):
                u = order[k]
                l = pline[u]
                loss += r[l] * (abs(Ib[u]) ** 2)
        # root power (grid import / island reference output)
        # Ib[root] accumulates the sum of child currents (children added themselves to parent).  Recompute exactly:
        Iroot = 0.0 + 0.0j
        for k in range(s0 + 1, s1):
            u = order[k]
            if parent[u] == rt:
                Iroot += Ib[u]
        sroot = Vc[rt] * np.conj(Iroot)
        # root node's own load / generation
        sroot_total = sroot.real + P[rt]
        cap_root = grid_cap if grid else 0.0
        if not grid:
            for j in range(len(pv_node)):
                if pv_node[j] == rt:
                    cap_root = pv_avail[j]
        # for islands the reference PV also carries its dispatch share
        lim = cap_root
        slack_here = lim * 1.0005 + 1e-6 - sroot_total
        if slack_here < supply_slack:
            supply_slack = slack_here
        if slack_here < 0.0:
            code = 3
        for k in range(s0, s1):
            u = order[k]
            V_out[u] = abs(Vc[u])
            va = abs(Vc[u])
            if va < vmin:
                vviol += vmin - va
            elif va > vmax:
                vviol += va - vmax
            if k > s0:
                l = pline[u]
                sb = abs(Vc[parent[u]] * np.conj(Ib[u]))
                S_out[l] = sb
                if sb > smax[l]:
                    sviol += sb - smax[l]
    if code == 0 and vviol > 1e-9:
        code = 4
    elif code == 0 and sviol > 1e-9:
        code = 5
    return code, vviol, sviol, supply_slack


@njit(cache=True)
def recoverable_goals(N, L, f, t, closed_all, Y, goals_out):
    """Nodes not energised that are adjacent (via any non-faulted line, switches assumed closable) to an energised node.
    closed_all[l] = line exists (not faulted).  Returns count."""
    n = 0
    for i in range(N):
        goals_out[i] = False
    for l in range(L):
        if not closed_all[l]:
            continue
        a, b = f[l], t[l]
        if Y[a] and (not Y[b]):
            if not goals_out[b]:
                goals_out[b] = True; n += 1
        elif Y[b] and (not Y[a]):
            if not goals_out[a]:
                goals_out[a] = True; n += 1
    return n


@njit(cache=True)
def feasible_pairs_ref(N, L, f, t, r, x, smax, adj_ptr, adj_line, adj_nb, closed0, Y0, P, Q, slack, grid_cap, vslack,
                   pv_node, pv_avail, mt_node, mt_cap, vmin, vmax, sw_line, edits, exist, out_edit, out_goal, out_key):
    """Enumerate (switch-edit, goal) pairs that are feasible.  edits[e, :2] are switch indices flipped (-1 = none).
    exist[l] = line not faulted.  Returns the number of pairs written.  Goals are the non-energised nodes joined to the
    energised set by a closed line under the edited configuration."""
    V = np.zeros(N); S = np.zeros(L)
    closed = closed0.copy()
    Pn = P.copy(); Qn = Q.copy()
    Y = Y0.copy()
    n = 0
    for e in range(edits.shape[0]):
        for k in range(2):
            s = edits[e, k]
            if s >= 0:
                closed[sw_line[s]] = not closed[sw_line[s]]
        # a switch cannot be closed over a faulted line
        bad = False
        for l in range(L):
            if closed[l] and not exist[l]:
                bad = True
        if not bad:
            # cheap necessary check: restored nodes keep supply -> evaluate baseline without goal first
            for l in range(L):
                if closed[l] and (Y[f[l]] != Y[t[l]]):
                    g = t[l] if Y[f[l]] else f[l]
                    Y[g] = True
                    code, vv, sv, ss = eval_state(N, L, f, t, r, x, smax, adj_ptr, adj_line, adj_nb, closed, Y, Pn, Qn,
                                                  slack, grid_cap, vslack, pv_node, pv_avail, mt_node, mt_cap,
                                                  vmin, vmax, V, S)
                    if code == 0:
                        # canonical key: goal + states of the switches whose both ends are energised (others are irrelevant now)
                        rel = 0; stt = 0
                        for sidx in range(sw_line.shape[0]):
                            ll = sw_line[sidx]
                            if Y[f[ll]] and Y[t[ll]]:
                                rel |= (1 << sidx)
                                if closed[ll]:
                                    stt |= (1 << sidx)
                        key = (np.int64(g) << 40) | (np.int64(rel) << 20) | np.int64(stt)
                        dup = False
                        for q in range(n):
                            if out_key[q] == key:
                                dup = True; break
                        Y[g] = False
                        if not dup and n < out_edit.shape[0]:
                            out_edit[n] = e; out_goal[n] = g; out_key[n] = key; n += 1
                        continue
                    Y[g] = False
        for k in range(2):
            s = edits[e, k]
            if s >= 0:
                closed[sw_line[s]] = not closed[sw_line[s]]
    return n


@njit(cache=True)
def feasible_pairs_pruned(N, L, f, t, r, x, smax, adj_ptr, adj_line, adj_nb, closed0, Y0, P, Q, slack, grid_cap, vslack,
                   pv_node, pv_avail, mt_node, mt_cap, vmin, vmax, sw_line, edits, exist, out_edit, out_goal, out_key):
    """Enumerate (switch-edit, goal) pairs that are feasible.  edits[e, :2] are switch indices flipped (-1 = none).
    exist[l] = line not faulted.  Returns the number of pairs written.  Goals are the non-energised nodes joined to the
    energised set by a closed line under the edited configuration."""
    V = np.zeros(N); S = np.zeros(L)
    closed = closed0.copy()
    Pn = P.copy(); Qn = Q.copy()
    Y = Y0.copy()
    n = 0
    for e in range(edits.shape[0]):
        # edits that flip a switch with no energised end cannot change the energised sub-graph or the reachable goals; the canonical
        # representative of every pair is a lower-order edit that is enumerated earlier, so skipping them leaves the output unchanged
        skip = False
        for k in range(2):
            s = edits[e, k]
            if s >= 0:
                ll = sw_line[s]
                if not (Y0[f[ll]] or Y0[t[ll]]):
                    skip = True
        if skip:
            continue
        for k in range(2):
            s = edits[e, k]
            if s >= 0:
                closed[sw_line[s]] = not closed[sw_line[s]]
        # a switch cannot be closed over a faulted line
        bad = False
        for l in range(L):
            if closed[l] and not exist[l]:
                bad = True
        if not bad:
            # cheap necessary check: restored nodes keep supply -> evaluate baseline without goal first
            for l in range(L):
                if closed[l] and (Y[f[l]] != Y[t[l]]):
                    g = t[l] if Y[f[l]] else f[l]
                    Y[g] = True
                    code, vv, sv, ss = eval_state(N, L, f, t, r, x, smax, adj_ptr, adj_line, adj_nb, closed, Y, Pn, Qn,
                                                  slack, grid_cap, vslack, pv_node, pv_avail, mt_node, mt_cap,
                                                  vmin, vmax, V, S)
                    if code == 0:
                        # canonical key: goal + states of the switches whose both ends are energised (others are irrelevant now)
                        rel = 0; stt = 0
                        for sidx in range(sw_line.shape[0]):
                            ll = sw_line[sidx]
                            if Y[f[ll]] and Y[t[ll]]:
                                rel |= (1 << sidx)
                                if closed[ll]:
                                    stt |= (1 << sidx)
                        key = (np.int64(g) << 40) | (np.int64(rel) << 20) | np.int64(stt)
                        dup = False
                        for q in range(n):
                            if out_key[q] == key:
                                dup = True; break
                        Y[g] = False
                        if not dup and n < out_edit.shape[0]:
                            out_edit[n] = e; out_goal[n] = g; out_key[n] = key; n += 1
                        continue
                    Y[g] = False
        for k in range(2):
            s = edits[e, k]
            if s >= 0:
                closed[sw_line[s]] = not closed[sw_line[s]]
    return n


feasible_pairs = feasible_pairs_pruned
