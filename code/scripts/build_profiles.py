"""Build per-day PV capacity-factor and load-shape arrays from the raw data folder.
Usage (run from the repository root):  python code/scripts/build_profiles.py <data_folder>
<data_folder> must contain '2012-2013 Solar home electricity data v2.csv/2012-2013 Solar home electricity data v2.csv' (Ausgrid) and 'isone_load.csv' (ISO-NE hourly load).
Outputs data/profiles/profiles.npz with arrays: dates, pv_cf (D,48), load_pu (D,24), split (D,) in {0,1,2}.
Split: blocked-temporal inside each calendar month (first 60% of days train, next 20% val, last 20% test)."""
import numpy as np, pandas as pd, sys, os
if len(sys.argv) < 2:
    sys.exit(__doc__)
root = sys.argv[1]
os.makedirs("data/profiles", exist_ok=True)
solar = os.path.join(root, "2012-2013 Solar home electricity data v2.csv", "2012-2013 Solar home electricity data v2.csv")
df = pd.read_csv(solar, skiprows=1)
gg = df[df["Consumption Category"] == "GG"].copy()
tcols = [c for c in gg.columns if ":" in c]
tcols = tcols[:48]
gg["date"] = pd.to_datetime(gg["date"], format="%d/%m/%Y")
vals = gg[tcols].to_numpy(float)
cap = gg["Generator Capacity"].to_numpy(float)
cf = 2.0 * vals / cap[:, None]  # kWh per 30 min -> kW / kW
cf = np.clip(cf, 0, 1.2)
gg_cf = pd.DataFrame(cf); gg_cf["date"] = gg["date"].values
daily = gg_cf.groupby("date").mean()
cnt = gg_cf.groupby("date").size()
daily = daily[cnt >= 5]
# 'tcols' are labelled 0:30..0:00(end): index i is the interval ending at (i+1)*30 min.
iso = pd.read_csv(os.path.join(root, "isone_load.csv"), sep=";")
iso["ts"] = pd.to_datetime(iso["date"], format="%d/%m/%Y") + pd.to_timedelta(iso["hour"] - 1, unit="h")
pjm = iso.set_index("ts")["load"].astype(float)   # ISO-NE hourly load (name kept for brevity)
pjm = pjm[(pjm.index >= "2012-06-30") & (pjm.index <= "2013-07-02")]
peak = pjm.max()
dates, pvs, loads = [], [], []
for d, row in daily.iterrows():
    day = pjm[(pjm.index >= d) & (pjm.index < d + pd.Timedelta(days=1))]
    if len(day) < 24:
        continue
    l = (day.groupby(day.index.hour).mean().reindex(range(24)).to_numpy()) / peak
    if np.isnan(l).any():
        continue
    dates.append(d); pvs.append(row.to_numpy(float)); loads.append(l)
dates = pd.DatetimeIndex(dates)
split = np.zeros(len(dates), int)
for (y, m), idx in pd.Series(np.arange(len(dates))).groupby([dates.year, dates.month]):
    idx = idx.to_numpy(); n = len(idx)
    a, b = int(round(0.6 * n)), int(round(0.8 * n))
    split[idx[a:b]] = 1; split[idx[b:]] = 2
np.savez("data/profiles/profiles.npz", dates=dates.strftime("%Y-%m-%d").to_numpy(), pv_cf=np.array(pvs), load_pu=np.array(loads), split=split)
print("days", len(dates), "split counts", np.bincount(split))
print("noon cf mean/min/max", np.mean([p[23] for p in pvs]), np.min([p[23] for p in pvs]), np.max([p[23] for p in pvs]))
print("load pu noon mean", np.mean([l[12] for l in loads]), "min", np.min([l[12] for l in loads]), "max", np.max([l[12] for l in loads]))
