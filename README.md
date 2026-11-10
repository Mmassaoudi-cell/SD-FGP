# SD-FGP: Search-Distilled, Feasibility-Screened Graph Policies for Sequential Distribution Service Restoration

Reference implementation of SD-FGP, a restoration controller for distribution feeders that

1. **screens** every candidate action exactly: for each node that can be energized next, the pairs (goal node, switch edit of at most two flips) are checked with a backward/forward-sweep power flow (radiality, reference source, supply capacity, voltage and branch limits, monotone restoration), so the policy only ever chooses among feasible actions;
2. **ranks** the screened pairs with an edge-conditioned graph neural network whose weights do not depend on the feeder size;
3. **distils** simulator-based beam-search plans into the ranking by behaviour cloning, and
4. **fine-tunes** the ranking with PPO under a behaviour-cloning regulariser.

The repository contains the model, the screening code and simulator it relies on, and the scripts needed to build scenarios, generate expert plans, train and evaluate SD-FGP. Baselines, experiment drivers, raw results and the manuscript are not part of this repository.

## Repository layout

```
code/dsr/        simulator and exact screening
  network.py       IEEE 33-bus (pandapower case33bw) and 120-node single-phase equivalent of the IEEE 123-node feeder
  core.py          numba power flow (backward/forward sweep), feasibility checks, canonical (goal, switch-edit) pair enumeration
  env.py           restoration environment: scenarios, monotone restoration, rewards, service-level integral, energy not served
  planners.py      greedy, rollout, beam-search and receding-horizon beam planners (model-based references and expert generator)
code/agents/     SD-FGP
  fgp.py           PairNet: edge-conditioned GNN encoder + pair-scoring and value heads; PPO agent over screened pairs
  fgp_ext.py       FGPDistill (SD-FGP): expert data loading, behaviour cloning, PPO with BC regulariser; value-guided look-ahead
  common.py        training loop with validation-based checkpoint selection, evaluation, metrics
  registry.py      method names -> factories (SD-FGP and its ablation variants) and planner wrappers
code/scripts/
  build_profiles.py   PV / load day profiles from the raw data
  make_manifest.py    fixed validation / test scenario pools (hash-based split of fault sets, blocked split of days)
  make_expert.py      beam-search expert plans on training scenarios
  run_one.py          train one method / system / seed and evaluate it on the chosen pools
  eval_ckpt.py        evaluate a saved checkpoint (optionally with perturbations or zero-shot transfer to the other feeder)
```

## Installation

Python 3.10 or newer. The code was run with Python 3.13, `numpy 2.4`, `scipy 1.16`, `numba 0.65`, `torch 2.x`, `pandas 2.3`, `pandapower 3.3` and `networkx 3.6`:

```
pip install numpy scipy numba torch pandas pandapower networkx
```

## Data

| Input | Source | Location |
|---|---|---|
| IEEE 33-bus network | included in `pandapower` (`case33bw`) | none needed |
| IEEE 123-node feeder | OpenDSS IEEE 123 Test Feeder (`IEEE123Master.dss`, `IEEELineCodes.DSS`, `IEEE123Loads.DSS`, `IEEE123Regulators.DSS`, `BusCoords.dat`) | `data/ieee123/` |
| PV capacity factors | Ausgrid "Solar home electricity data" 2012-2013 | passed to `build_profiles.py` |
| Load shape | ISO-NE hourly load (`isone_load.csv`, columns `date`, `hour`, `load`) | passed to `build_profiles.py` |

Training scenarios are streamed from the training partition (fault sets are assigned to train / validation / test by a hash of the sorted fault tuple, 70/15/15 %; days are split in blocks inside each month, 60/20/20 %), so no fault set or day is shared between partitions.

## Usage

Run all commands from the repository root.

```
# 1. day profiles and fixed evaluation pools (writes data/profiles/profiles.npz, data/pools_<system>.pkl, DATA_SPLIT_MANIFEST.csv and results/split_audit.json)
python code/scripts/build_profiles.py /path/to/raw_data_folder
python code/scripts/make_manifest.py

# 2. beam-search expert plans on training scenarios: <system> <n_scenarios> <beam_width> <n_processes> [tag]
python code/scripts/make_expert.py ieee33 1500 24 8                 # -> data/expert_ieee33.pkl
python code/scripts/make_expert.py ieee123 1500 16 8 _v2            # -> data/expert_v2_ieee123.pkl

# 3. train SD-FGP (best validation checkpoint is kept; test pools are refused unless DSR_ALLOW_TEST=1)
python code/scripts/run_one.py --method fgp_distill --system ieee33 --seed 0 --episodes 800 --eval_every 100 \
    --pools val_dev --out results/sdfgp --save_ckpt \
    --cfg '{"lr":0.00039468221841458106,"H":96,"layers":3,"ent":0.0022836507782347535,"clip":0.21338626863438998,"epochs":2,"rollout_steps":1024,"gamma":0.9732433554985835,"bc_w":0.1594055559916556,"bc_epochs":9}'

#    123-node feeder: 600 episodes and the 1500-plan expert data
python code/scripts/run_one.py --method fgp_distill --system ieee123 --seed 0 --episodes 600 --eval_every 100 \
    --pools val_dev --out results/sdfgp --save_ckpt \
    --cfg '{"lr":0.00039468221841458106,"H":96,"layers":3,"ent":0.0022836507782347535,"clip":0.21338626863438998,"epochs":2,"rollout_steps":1024,"gamma":0.9732433554985835,"bc_w":0.1594055559916556,"bc_epochs":9,"expert_files":["expert_v2_ieee123.pkl"]}'

# 4. evaluate a saved checkpoint on the final test pools (frozen model only)
DSR_ALLOW_TEST=1 python code/scripts/eval_ckpt.py --run results/sdfgp/fgp_distill_ieee33_s0 --pools test --out results/sdfgp_eval
```

Model-based references use the same interface: `--method greedy | rollout | beam | mpc` (`mpc` is the non-anticipative receding-horizon beam search; `--cfg '{"width":8,"horizon":5}'`).

Each run writes `<out>/<method>_<system>_s<seed>.json` with per-scenario arrays (service restoration rate, complete-restoration indicator, service-level integral, energy not served, infeasible attempts, decision latency when `--time_decisions` is given) and, with `--save_ckpt`, the best and final checkpoints.

## Metrics

* **Service-level integral (SLI)**: time-averaged importance-weighted share of restored load over the episode horizon (|outage nodes| + 3 steps of 3 minutes).
* **SRR / CRR**: final weighted service restoration rate and the share of scenarios restored completely.
* **ENS**: energy not served (kWh) of the interrupted nodes over the horizon.
* **Infeasible attempts** per episode (zero for every screened controller by construction).

## Notes

* The 33-bus configuration follows the source study's switch and DER layout; the 123-node system is a balanced single-phase equivalent of the OpenDSS IEEE 123-node feeder, not the modified system of that study.
* The numba kernels are compiled on first use and cached; the first call of a new session is slower.
* Results depend on the random seeds, the PyTorch and numba versions and on thread scheduling; run-to-run differences are of the order of the seed standard deviations reported in the manuscript.

## Citation

The accompanying manuscript is under preparation; the citation will be added here when it is available.

## License

See [LICENSE](LICENSE).
