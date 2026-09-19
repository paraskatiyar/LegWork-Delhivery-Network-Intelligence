# Technical Appendix

*Generated 10 September 2026 from the pipeline artifacts in `outputs/`.*

---

## 1. From scan records to delivery legs

The raw file is at scan grain. `actual_time` is cumulative for a leg and is therefore repeated on every scan row belonging to that leg. Modelling the raw rows means fitting and scoring against duplicated targets, and the resulting error metrics are not comparable to leg-level ones.

There are two ways to reconstruct a leg, and they disagree:

| Reconstruction | Legs recovered | Note |
|---|---|---|
| `is_cutoff == False` summary rows | 26,117 | Misses 251 legs that have no summary row; 1 duplicate key |
| Sum of `segment_actual_time` | 26,368 | Complete, but disagrees with the summary row on 65% of legs |

We take the summary row where it exists and fall back to the segment sum otherwise, keeping all 26,368 legs. The two agree exactly on 34.9% of legs; the median disagreement is 1 minute and the 95th percentile is 7 minutes, so the choice is immaterial to conclusions but is recorded rather than assumed away. Every row carries a `reconstruction_source` column.

Split: 18,947 training legs and 7,421 holdout legs, using the dataset's own `data` column rather than a random split. A random split over scan rows would put segments of the same shipment on both sides of the partition.

## 2. Defining lateness

| Definition | Late rate | Usable for ranking? |
|---|---|---|
| raw: actual > 1.2x OSRM | 94.8% | no - fires on almost every leg |
| calibrated: actual > corridor promise x 1.15 (promise = OSRM x shrunk corridor median, train-fitted) | 26.1% | yes - concentrates on genuine operational excess |

The brief's 1.2x rule flags 95% of legs. It conflates two distinct quantities. Decomposing the total delay against OSRM:

- **Systematic forecast bias: 78.2%** (2,271,338 minutes). OSRM models free-flow driving with no facility dwell. Predictable, and fixed by a better model.
- **Within tolerance: 7.4%** (215,108 minutes). The 15% band between the promise and the late threshold. Not a failure of anything.
- **Operational excess: 14.4%** (418,991 minutes). Variation beyond what the corridor normally achieves. The only part a facility upgrade addresses.

The three partition the gap exactly and a test enforces it. Computing them independently -- the obvious implementation -- sums past 100%, because the tolerance band belongs to neither of the other two and because a leg that beats its promise still books a full systematic share it never actually incurred. Each leg's systematic component is therefore capped at its own realised gap.

The calibrated promise is `OSRM x shrunk corridor median delay ratio`, fitted on training legs only, with empirical-Bayes shrinkage (k=5) toward the network median so that a corridor seen twice does not define its own target. A leg is late when it exceeds that promise by more than 15%, matching the accuracy band the brief uses for ETA.

## 3. ETA benchmark

Every rung uses the identical model class (LightGBM, L1 objective), hyperparameters, target and split. Rungs are strictly nested: each adds one block of features to the previous one. A difference between adjacent rungs is therefore attributable to that block.

| Rung | Information added | MAE (min) | Within 15% | Within 25% | Bias (min) |
|---|---|---|---|---|---|
| `osrm` | OSRM raw estimate (incumbent, no model) | 107.78 | 4.3% | 9.6% | -107.4 |
| `trip` | Tabular baseline (routing estimate + calendar) | 41.28 | 43.7% | 62.6% | -19.2 |
| `corridor` | + corridor history | 32.17 | 56.2% | 72.8% | -9.7 |
| `structural` | + network position (centrality) | 31.99 | 56.1% | 73.6% | -9.8 |
| `embedded` | + learned node embeddings | 31.95 | 56.7% | 73.2% | -10.1 |

### What each block is worth

| Step | Question | MAE reduction | 95% CI | Within-15% gain | Significant |
|---|---|---|---|---|---|
| `osrm -> trip` | Does any model beat the routing engine we ship today? | +66.50 min | [+62.65, +70.60] | +39.4pp | yes |
| `trip -> corridor` | Does knowing this lane's own history help? | +9.11 min | [+7.46, +10.77] | +12.6pp | yes |
| `corridor -> structural` | Does network position add beyond lane history? | +0.18 min | [+0.01, +0.35] | -0.1pp | yes |
| `structural -> embedded` | Do learned embeddings add beyond hand-built metrics? | +0.04 min | [-0.25, +0.33] | +0.6pp | **no** |
| `trip -> embedded` | Total graph advantage over a non-graph model. | +9.33 min | [+7.55, +11.00] | +13.0pp | yes |

Intervals are 95% paired bootstraps over per-leg absolute errors (2,000 resamples). Pairing matters: both models saw the same legs, so the correlated component of the error cancels.

## 4. Negative results

These are reported because they are the most useful findings in the project, and because the temptation to bury them is exactly why graph-ETA work tends to report advantages that do not replicate.

### 4.1 Learned node embeddings add nothing here

| Embedder | MAE at `embedded` | Gain over `structural` | 95% CI | Significant |
|---|---|---|---|---|
| graphsage | 31.84 min | +0.15 min | [-0.06, +0.36] | **no** |
| node2vec | 31.95 min | +0.04 min | [-0.25, +0.33] | **no** |
| svd | 32.16 min | -0.18 min | [-0.45, +0.09] | **no** |

All three representations -- node2vec (biased walks + skip-gram), GraphSAGE (unsupervised mean-aggregator), and truncated SVD of the adjacency matrix -- fail to improve on hand-built corridor history plus centrality features. The result replicates across methods, so it is a property of the problem rather than of one implementation.

The likely reason: with ~1,350 facilities and ~1,750 corridors, the corridor's own median delay ratio already captures nearly everything the topology has to say about that corridor. Embeddings would earn their keep on a denser network, or for corridors with no history -- which is precisely the cold-start case below, and worth revisiting there.

### 4.2 Network position adds little beyond corridor history

Adding betweenness, PageRank, clustering and degree for both endpoints improves MAE by 0.18 minutes [+0.01, +0.35] and moves within-15% by -0.1pp. Statistically detectable, operationally negligible.

**This does not make the graph useless.** It relocates its value. The graph earns its keep as a *memory of lanes* -- corridor history is worth 9.1 minutes and 12.6 percentage points -- and as the structure that makes the bottleneck audit and the hub prioritisation possible at all. Those are graph products, and they are what the memo actually recommends acting on. The graph is not, on this data, a better function approximator.

## 5. Leakage controls

Enforced mechanically, not by convention:

1. `NetworkModel.fit()` raises if handed anything other than training legs. The graph, its edge statistics, the centrality metrics and the SLA calibration are all fitted on the training split and applied unchanged to the holdout.
2. Node embeddings are fitted on the training graph only, so no test-set corridor contributes to any facility's representation.
3. `assert_no_leakage()` aborts the run if any column in `FORBIDDEN_FEATURES` reaches a model matrix. That list covers everything known only after the shipment completes: `actual_time`, `factor`, `segment_*`, `start_scan_to_end_scan`, `cutoff_factor`, `is_cutoff`, `actual_distance_to_destination`.
4. `assert_units()` fails the run if the median leg duration leaves the 10-600 minute band, catching minute/second/hour confusions before they reach a document.

The exclusions in point 3 are worth defending individually. `start_scan_to_end_scan` is the wall-clock leg duration and correlates 0.95 with the target; `cutoff_factor` is post-hoc scan bookkeeping; `actual_distance_to_destination` is remaining distance, known only in transit. Each is available in the file and each would improve the reported metric. None is available at the moment an ETA has to be quoted, which is the only moment that matters.

## 6. Cold start

Of 7,421 holdout legs, 6.1% touch a facility absent from the training graph and 11.8% run on a corridor with no training history (135 unseen source facilities, 141 unseen destinations).

These legs carry `edge_is_cold_start` / `src_is_cold_start` / `dst_is_cold_start` flags into the model, so it can learn to fall back rather than extrapolate from a mean embedding it should not trust. In production the fallback rate is the metric to monitor: it measures how fast the network is outgrowing the model's last retrain.

## 7. FTL vs Carting

The counterfactual is model-based: each leg is scored twice, once per mode, holding distance, hour and network position fixed. Recommendations are restricted to the 25-86 km band where both modes are actually observed (57% of legs); outside it the comparison would be extrapolation.

Within that band FTL buys a median of 0.16 minutes for a mean premium of Rs 1,349. That is the whole result: on short lanes FTL is not meaningfully faster, so it cannot repay its cost. Today 47% of these legs run FTL.

### Break-even distance

| Late penalty (Rs/min) | FTL break-even (generalised) | Transport cost only |
|---|---|---|
| 17.5 | 179 km | 180 km |
| 26.2 | 179 km | 180 km |
| 35.0 | 179 km | 180 km |
| 52.5 | 179 km | 180 km |
| 70.0 | 179 km | 180 km |

The break-even barely moves with the lateness penalty, because the time FTL buys on these lanes is close to zero -- there is almost nothing for a higher penalty to multiply. The decision on short haul is pure haulage economics.

**Empirical cross-check.** Only 23 corridors ever ran both modes, which is far too few to decide policy on and is why the counterfactual is model-based. On those corridors FTL was faster on 52% -- effectively a coin flip, consistent with the model's finding of no material time advantage at these distances.

**Every cost parameter is invented.** The dataset has no cost column. The rate card lives in `configs/pipeline.yaml`, is printed with the results, and is swept in `outputs/stats/routing_sensitivity.csv`. Conclusions that do not survive that sweep are flagged as such.

## 8. Limitations

- **24 days of data, September-October 2018.** Enough for corridor-level medians, not enough for seasonality. Any annualised figure in the memo assumes the sampled run-rate holds, and this window includes festive volume.
- **This is a sample of the network, not the network.** Absolute rupee figures are small by construction. Per-100k-leg normalisations are provided so the reader can scale by true volume rather than have us invent one.
- **The counterfactual is predictive, not causal.** Mode is confounded with distance and corridor. The overlap-band restriction limits the damage but does not eliminate it; a genuine answer needs a dispatch experiment.
- **The chokepoint score's weights are a judgement call** (0.35 betweenness, 0.20 in-volume, 0.20 out-volume, 0.25 excess contribution). The top three hubs are stable under reweighting; ranks 4-8 are not.
- **Facility-level SLA attribution double-counts by construction**, since a leg touches two facilities. Shares are computed against total attributions so the column sums to 100%, but a hub's share is not a share of legs.
- **2,674 corridors** exceed the 1.2x threshold and only 99 carry enough volume to act on. The rest are reported, not ranked.

## 9. Reproducing this

```bash
pip install -r requirements.txt
python run.py all                      # full pipeline
python run.py model --embedder graphsage   # swap the representation
python -m pytest tests -q              # correctness guards
```

Every number in the memo and in this appendix is written by the pipeline into `outputs/stats/` as both parquet and CSV. Changing an assumption in `configs/pipeline.yaml` and re-running regenerates both documents with it.
