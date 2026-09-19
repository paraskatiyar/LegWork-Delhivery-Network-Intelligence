# Delhivery Network Intelligence

Graph-based ETA prediction, bottleneck detection, and route-type economics for a
hub-and-spoke logistics network.

```bash
pip install -r requirements.txt
# place delivery_data.csv in data/raw/
python run.py all --config configs/pipeline.yaml     # ~90 seconds
python -m pytest tests -q                            # 18 correctness guards
```

Outputs land in `outputs/`: an operations memo, a technical appendix, eight
figures, and every intermediate table as both parquet and CSV.

---

## The three findings

**1. Most of the "delay" is a forecasting problem, not an operations problem.**

The usual measure — actual time vs. OSRM + 20% — flags **95%** of legs as late.
A metric that fires on nineteen of every twenty shipments cannot prioritise
anything. Decomposing the gap against OSRM into its parts:

| Component | Share | What fixes it |
|---|---:|---|
| Systematic forecast bias | **78.2%** | A better model. Near-zero capex. |
| Within tolerance of the promise | 7.4% | Nothing; it is inside the promise. |
| Genuine operational excess | **14.4%** | Facility and corridor work. |

(The three partition the gap exactly — a test enforces it. Computing them
independently, as is tempting, sums past 100% because the tolerance band belongs
to neither bucket and legs that beat their promise still book a systematic share
they never incurred.)

OSRM models free-flow driving with no facility dwell, so it is optimistic by
construction — its median leg estimate is ~1.9× short. Against a promise
calibrated to what each corridor actually achieves, the real late rate is **26%**.
This reordering is the project's main analytical result: it says ship the model
before pouring concrete, and it caps what a capex case can honestly claim.

**2. The graph advantage is real, moderate, and not where you would expect.**

| Rung | Information added | MAE (min) | Within 15% |
|---|---|---:|---:|
| `osrm` | The incumbent routing engine | 107.78 | 4.3% |
| `trip` | Routing estimate + calendar. No graph. | 41.28 | 43.7% |
| `corridor` | + this lane's own history | 32.17 | 56.2% |
| `structural` | + centrality of both endpoints | 31.99 | 56.1% |
| `embedded` | + node2vec embeddings | 31.95 | 56.7% |

Total graph advantage over a non-graph model: **−9.3 minutes MAE (−23%), +13.0
percentage points** within 15%, with a 95% paired-bootstrap interval of
[7.6, 11.0] minutes. Real, and comfortably clear of zero.

But almost all of it — 9.11 of the 9.33 minutes — comes from **corridor history**.
Centrality adds 0.18 minutes. Learned embeddings add nothing:

| Embedder | Gain over hand-built features | 95% CI | Significant |
|---|---:|---|---|
| GraphSAGE | +0.14 min | [−0.08, +0.35] | no |
| node2vec | +0.04 min | [−0.25, +0.33] | no |
| SVD | −0.18 min | [−0.45, +0.09] | no |

Three independent representations all fail to beat a median and a betweenness
score. That replicates across methods, so it is a property of this network
(1,353 facilities, 1,751 corridors), not of one implementation. The graph earns
its keep as a **memory of lanes** and as the structure that makes the bottleneck
audit possible — not as a better function approximator.

**3. On short lanes, FTL does not buy time.**

In the 25–86 km band where both modes actually operate, FTL is a median of
**0.2 minutes** faster and costs **₹1,349** more per shipment. On the stated rate
card it only repays its premium beyond ~**180 km**, and that threshold barely
moves even at 4× the lateness penalty — there is almost no time saving for a
higher penalty to multiply. Today **47%** of legs in that band run FTL.

---

## What this project does differently

This was built after reviewing five existing solutions to the same brief. Each
correction below addresses a specific failure observed in that review.

**Leg reconstruction is reconciled, not assumed.** The file is at scan grain and
`actual_time` is cumulative per leg, so it repeats across every scan of that leg.
Two reconstructions exist and they disagree: the `is_cutoff == False` summary
rows recover 26,117 legs but silently drop 251 that have no summary row (and
carry one duplicated key), while summing `segment_actual_time` recovers all
26,368 but disagrees with the summary on 65% of legs (median 1 minute). We take
the summary where it exists, fall back to the segment sum, keep every leg, and
record the disagreement as a data-quality artifact.

**Leakage is prevented structurally, not by discipline.** `NetworkModel.fit()`
raises if handed anything but training legs. Embeddings and SLA calibration are
train-fitted. `assert_no_leakage()` aborts the run if any outcome-derived column
reaches a model matrix — `actual_time`, `factor`, `segment_*`,
`start_scan_to_end_scan`, `cutoff_factor`, `actual_distance_to_destination`.
Each of those is present in the file and each would improve the reported metric;
none is available at the moment an ETA has to be quoted.

**The benchmark is an ablation ladder.** Every rung uses the identical model
class, hyperparameters, target and split, and rungs are strictly nested, so a
gap between two rungs is attributable to the block that was added. Handing a
graph model corridor history while denying it to the baseline measures the
feature, not the graph.

**Improvements carry uncertainty.** Every claimed gain reports a 95% paired
bootstrap over per-leg absolute errors. Two of the five ladder steps turn out
not to clear zero, and they are reported as such.

**Units are asserted.** `assert_units()` fails the run if the median leg
duration leaves the 10–600 minute band, catching the minute/second/hour
confusions that otherwise surface as an "MAE of 12.55 hours" in a finished
document.

**Corridor priorities require volume.** Ranking by delay ratio surfaces lanes
running ten shipments where one mis-scan produces a 26× reading. Priorities are
ranked by contributed excess minutes, with per-leg excess capped at the 99th
percentile and a 30-leg floor. The 112 artifact corridors are routed to a
data-quality review instead.

**Route choice is a counterfactual, not a classifier.** Predicting which mode
*was* used scores ~99% and tells you only that the existing policy is
consistent. We score each leg twice — once per mode — price both through a
stated cost model, and restrict recommendations to the distance band where both
modes are actually observed. Only 23 corridors ever ran both modes, which is why
the counterfactual has to be model-based; those 23 serve as a sign check.

**Cost and revenue assumptions are declared, swept, and never sourced to an
invented benchmark.** The dataset has no cost column. Every parameter sits in
`configs/pipeline.yaml`, is printed with the results, and is swept in a
sensitivity analysis. Figures are reported as ranges, and normalised per 100k
legs so a reader can scale by true volume rather than have us invent one.

**The memo is generated from the artifacts.** Prose and numbers come from the
same frames, so an executive summary cannot disagree with its own supporting
table. Change an assumption and re-run: the memo changes with it.

**Two documents, two audiences.** The memo names hubs in plain English and
contains no MAE, no embeddings, no architecture. The appendix carries the
method, the benchmark, the negative results and the limitations.

---

## Layout

```
dni/
  config.py       all tunables; loads configs/pipeline.yaml
  ingest.py       scan rows -> legs, reconciliation, unit assertions, leakage list
  sla.py          calibrated promise, three-way delay decomposition
  graph.py        directed weighted graph, centrality suite, corridor audit
  embeddings.py   node2vec, GraphSAGE, SVD -- implemented directly on numpy/torch
  features.py     nested feature blocks + the leakage firewall
  models.py       ablation ladder, metrics, paired bootstrap
  routing.py      FTL/Carting counterfactual, break-even, sensitivity
  economics.py    revenue at risk, hub upgrade simulation
  viz.py          eight figures
  memo.py         operations memo, generated
  appendix.py     technical appendix, generated
run.py            CLI: all | prep | audit | model | routing | memo | compare
tests/            18 guards, each encoding an observed failure mode
```

`node2vec` and `GraphSAGE` are implemented here rather than imported. That keeps
the dependency surface to scikit-learn, LightGBM and torch, makes the
second-order walk bias and the mean-aggregation step readable, and lets both be
fitted strictly on the training graph.

## Stages

| Command | Does | Time |
|---|---|---|
| `python run.py prep` | Legs, SLA calibration, graph | ~11s |
| `python run.py audit` | Bottleneck + corridor audit, 4 figures | ~23s |
| `python run.py model` | Ablation ladder, 3 figures | ~50s |
| `python run.py routing` | FTL/Carting, break-even, sensitivity | ~15s |
| `python run.py memo` | Economics, memo, appendix | ~5s |
| `python run.py compare` | All three embedders, head to head | ~100s |

`--embedder {node2vec,graphsage,svd}` swaps the representation.

## Known limitations

- **24 days of data (Sep–Oct 2018).** Enough for corridor medians, not for
  seasonality. Annualised figures assume the sampled run-rate holds, and the
  window includes festive volume.
- **A sample of the network, not the network.** Absolute rupee figures are small
  by construction; per-100k-leg normalisations are provided.
- **The counterfactual is predictive, not causal.** Mode is confounded with
  distance and corridor. The overlap-band restriction limits the damage but does
  not remove it; a real answer needs a dispatch experiment.
- **Chokepoint weights are a judgement call.** The top three hubs are stable
  under reweighting; ranks 4–8 are not.
- **Facility SLA attribution double-counts**, since a leg touches two
  facilities. Shares are computed against total attributions so the column sums
  to 100%, but a hub's share is not a share of legs.
