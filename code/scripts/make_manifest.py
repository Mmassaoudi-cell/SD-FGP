"""Generate the fixed validation / test scenario pools and DATA_SPLIT_MANIFEST.csv; audit cross-split overlap.
Pools: val_select (checkpoint selection, 60), val_dev (development comparisons, 200), test (final, 200) and
test_tod_{8,16,20} (time-of-day shift, 100 each).  Training scenarios are streamed from the 'train' partition."""
import os, sys, pickle, csv, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsr.env import DSREnv, fault_split, Scenario, profiles

OUT = os.path.join(os.path.dirname(__file__), "..", "..")
POOLS = [("val_select", "val", 60, 101, 12.0), ("val_dev", "val", 200, 102, 12.0), ("test", "test", 200, 201, 12.0),
         ("test_tod8", "test", 100, 202, 8.0), ("test_tod16", "test", 100, 203, 16.0), ("test_tod20", "test", 100, 204, 20.0)]


def build(system):
    env = DSREnv(system)
    n = env.net
    prof = profiles()
    rows, pools = [], {}
    for name, split, cnt, seed, hour in POOLS:
        rng = np.random.default_rng(seed + (0 if system == "ieee33" else 1000))
        out = []
        while len(out) < cnt:
            scn = env.sample_scenario(split, rng, hour=hour)
            if any(scn.faults == o.faults and scn.day == o.day for o in out):
                continue
            out.append(scn)
        pools[name] = out
        for i, s in enumerate(out):
            env._setup(s)
            rows.append(dict(system=system, pool=name, index=i, split=s.split,
                             faults_buses=";".join(f"{n.labels[n.f[l]]}-{n.labels[n.t[l]]}" for l in s.faults),
                             fault_lines=";".join(map(str, s.faults)), day_index=s.day, date=str(prof.dates[s.day]), hour=s.hour,
                             noise_seed=s.seed, n_outage_nodes=env.n_dead))
    # training scenario stream descriptor + overlap audit on a large sample of the training partition
    rng = np.random.default_rng(777)
    train_sets = set(); train_days = set()
    for _ in range(4000):
        s = env.sample_scenario("train", rng)
        train_sets.add(s.faults); train_days.add(s.day)
    leak_exact = 0; leak_days = 0; max_j = []
    for name, ps in pools.items():
        for s in ps:
            leak_exact += s.faults in train_sets
            leak_days += s.day in train_days
            best = 0.0
            sf = set(s.faults)
            for tf in train_sets:
                if len(tf) and len(sf):
                    j = len(sf & set(tf)) / len(sf | set(tf))
                    best = max(best, j)
                    if best == 1.0:
                        break
            max_j.append(best)
    audit = dict(system=system, train_sample_scenarios=4000, unique_train_fault_sets=len(train_sets), exact_fault_set_overlap=leak_exact,
                 shared_days_with_train=leak_days, mean_max_jaccard=float(np.mean(max_j)), frac_jaccard_ge_0_75=float(np.mean(np.array(max_j) >= 0.75)))
    return rows, pools, audit


if __name__ == "__main__":
    for d in ("data", "results"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    all_rows, audits = [], []
    for system in ("ieee33", "ieee123"):
        rows, pools, audit = build(system)
        all_rows += rows; audits.append(audit)
        pickle.dump({k: [(s.faults, s.day, s.hour, s.seed, s.split) for s in v] for k, v in pools.items()},
                    open(os.path.join(OUT, "data", f"pools_{system}.pkl"), "wb"))
        print(audit)
    with open(os.path.join(OUT, "DATA_SPLIT_MANIFEST.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    import json
    json.dump(audits, open(os.path.join(OUT, "results", "split_audit.json"), "w"), indent=1)
    print("rows", len(all_rows))
