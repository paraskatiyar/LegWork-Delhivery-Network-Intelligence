# Network Operations Strategy Memo

**To:** Head of Network Operations  
**From:** Data Science  
**Date:** 10 September 2026  
**Subject:** Where our delivery promises break, and the two things to fix first

---

## The short version

We analysed 26,368 delivery legs across 1,353 facilities. Three things came out of it, and the third changes where we should spend money.

**1. Our delivery promise is wrong far more often than our operation is late.** Measured the way the industry usually does it - actual time versus the routing engine's estimate plus 20% - 95% of legs look late. That number is useless: it fires on almost every shipment we run. When we instead ask whether a leg missed what that corridor reliably achieves, the real figure is **26%**. The difference is not a rounding argument. **78% of the gap against the routing engine is systematic forecast error** - the engine assumes clean traffic and no time spent inside our own buildings. Only 14% is genuine operational excess.

**2. That means the cheapest fix is software, not concrete.** Quoting the new model's estimate instead of the routing engine's takes the share of shipments that miss their quoted window from 96% to 43% - roughly 13,808 legs in the sampled period, worth Rs 6.2 lakh in the period and about Rs 93.8 lakh annualised. No facility changes.

**3. The operational excess that remains is concentrated.** The top five hubs below carry 16% of it. Halving excess delay at the top three recovers 5.5% of our late legs, worth Rs 16,965 in the period (Rs 2.6 lakh annualised), with a plausible range of Rs 1.0 lakh to Rs 4.3 lakh depending on how much excess an upgrade actually removes and what a missed promise costs us.

**Sequence the two levers.** Ship the model first: it is weeks, near-zero capital, and it addresses the 78% of the gap that no amount of dock space can touch. Then upgrade the hubs, which is the only lever that reaches the remaining 14%.

## The five hubs to fix, in order

Ranked by a chokepoint score combining how much of the network routes through the facility, how much traffic it handles, and how many excess minutes it contributes. All three matter: a hub can be central and fine, or slow and irrelevant. These five are neither.

| # | Facility | Share of network excess | Legs touched | Late rate | What to do |
|---|---|---|---|---|---|
| 1 | **Gurgaon Bilaspur HB (Haryana)** | 6.0% | 1,991 | 21% | Protect it rather than rebuild it. Throughput is high (1,991 legs) but the miss rate is a contained 21% - it sits on 14% of network paths, so the risk is contagion, not current failure. Stage a parallel route before volume grows into it. |
| 2 | **Bangalore Nelmngla H (Karnataka)** | 1.6% | 1,433 | 22% | Capacity and process together. It touches 1,433 legs and misses the promise on 22% of them: add dock capacity, then a second dispatch wave to use it. |
| 3 | **Bhiwandi Mankoli HB (Maharashtra)** | 3.2% | 1,404 | 30% | Capacity and process together. It touches 1,404 legs and misses the promise on 30% of them: add dock capacity, then a second dispatch wave to use it. |
| 4 | **Hyderabad Shamshbd H (Telangana)** | 1.5% | 734 | 18% | Watch and instrument. It ranks on structural position (8% of network paths) rather than on current failure. Add it to control-tower monitoring; no capital case yet. |
| 5 | **Kolkata Dankuni HB (West Bengal)** | 3.9% | 481 | 41% | Process, not concrete. Volume is moderate but 41% of legs touching it miss the promise - attack dwell time and dock scheduling first; capacity spend here would buy little. |

The capital case covers the first three: **Gurgaon Bilaspur HB (Haryana), Bangalore Nelmngla H (Karnataka), Bhiwandi Mankoli HB (Maharashtra)**.

## The corridors doing the damage

Ranked by how many excess minutes each contributes, not by how bad its ratio looks. That distinction matters: our worst ratios sit on lanes running ten shipments a month, where a single mis-scan produces a 25x reading. Those are a data-quality task, not an investment case, and they are excluded here.

| Corridor | Legs | Typical vs promise | Share of excess | Intervention |
|---|---|---|---|---|
| Delhi Mayapuri PC (Delhi) &rarr; Gurgaon Bilaspur HB (Haryana) | 39 | 2.08x | 1.6% | Short-haul: the delay is dwell at the ends, not transit. Fix dock scheduling. |
| Gurgaon Bilaspur HB (Haryana) &rarr; Chandigarh Mehmdpur H (Punjab) | 58 | 2.21x | 0.8% | Long-haul FTL lane: hold a guaranteed departure window and stage a mid-point relay. |
| Mumbai Chndivli PC (Maharashtra) &rarr; Bhiwandi Mankoli HB (Maharashtra) | 99 | 2.73x | 0.8% | Short-haul: the delay is dwell at the ends, not transit. Fix dock scheduling. |
| Sonipat Kundli H (Haryana) &rarr; Gurgaon Bilaspur HB (Haryana) | 86 | 1.91x | 0.6% | Stage a parallel route and shift departures out of the congested window. |
| Gurgaon Bilaspur HB (Haryana) &rarr; Delhi Jhilmil L (Delhi) | 36 | 1.98x | 0.6% | Short-haul: the delay is dwell at the ends, not transit. Fix dock scheduling. |

> **Data-quality flag.** 112 corridors show median times more than 6x the routing estimate on low volumes. These are almost certainly mis-scans, one-off disruptions, or lanes where the routing engine has the geometry wrong. They are excluded from the priority list above and should go to a separate data-quality review, not to capital planning.

## The promise itself

Today we quote the routing engine's number. It lands within 15% of the truth on **4%** of legs, and it is biased low by 107 minutes on average - it under-promises the duration on almost every shipment, which is why our customers experience us as chronically late.

Replacing it with a model that reads the network as a connected graph - each corridor's own history, and each facility's position in the network - lands within 15% on **57%** of legs, with an average error of 32 minutes against the routing engine's 108.

We checked whether the graph is doing real work rather than flattering itself. Against an equivalent model given the same shipment details but no network information, the graph version cuts average error by 9.3 minutes (23%), and the 95% confidence interval on that gap runs 7.5 to 11.0 minutes - comfortably clear of zero. It is a real effect and a moderate one. Anyone reporting a 40-80% improvement from graph features on this dataset has almost certainly let the answer leak into the question.

**Deployment caveat worth knowing before you sign off:** 6% of legs in our holdout period start or end at a facility the model has never seen, and 12% run on a corridor with no history. The network opens new facilities faster than a model retrains. Those shipments must fall back to the routing-engine estimate with a widened window, and the fallback rate should be on the monitoring dashboard from day one.

## FTL vs Carting

We priced both modes for every shipment - predicted duration under each, plus transport cost, the value of the time, and the penalty if it misses its window - and compared the totals. We only advise on the 57% of shipments in the distance range where both modes genuinely operate today. Outside that range the comparison is extrapolation, and we leave the current choice alone.

**The finding is blunt: on these lanes, FTL does not buy time.** In the 25-86 km range where we run both modes, FTL is a median of 0.2 minutes faster and costs a mean Rs 1,349 more per shipment. Today **47%** of those shipments go FTL. That is the largest single cost saving this analysis found, and it needs no new capability - only a dispatch rule.

**The rule itself is a distance.** On our rate card FTL only repays its premium beyond roughly **179 km**. That threshold barely moves even if we quadruple what we charge ourselves for a late delivery, because there is almost no time saving for a higher penalty to multiply. Below it, default to Carting; above it, FTL.

Applied across the band, the framework would change the mode on **49%** of shipments.

| Distance | Time of day | Source hub | Shipments | Today | Recommended | Saving per shipment |
|---|---|---|---|---|---|---|
| 75-150km | late_night | major | 23 | 78% FTL | Default Carting | Rs 1,665 |
| 25-75km | morning | major | 164 | 15% FTL | Default Carting | Rs 1,514 |
| 25-75km | night | peripheral | 44 | 82% FTL | Default Carting | Rs 1,098 |
| 25-75km | morning | peripheral | 165 | 64% FTL | Default Carting | Rs 918 |
| 25-75km | late_night | major | 258 | 51% FTL | Default Carting | Rs 917 |
| 25-75km | late_night | connector | 731 | 66% FTL | Default Carting | Rs 903 |

**The honest caveat.** We have no cost data - the rate card behind these numbers is our assumption, stated in the configuration file and swept across a range. The *direction* survives that sweep: Carting short, FTL long, and short-haul FTL never justified on any setting we tried. The *exact* crossover distance does not survive it - it moves with the haulage rates, which are the numbers we are least sure about. Before acting, replace our rate card with the real one and re-run; it is one configuration change and the whole analysis, including this memo, regenerates.

## What this is worth

Revenue at risk in the sampled period, at our central assumption of Rs 450 per leg and a 10% penalty on a missed promise: **Rs 3.1 lakh** across 6,873 late legs.

| Lever | Period | Annualised | What it depends on |
|---|---|---|---|
| Ship the graph ETA model | Rs 6.2 lakh | Rs 93.8 lakh | Nothing physical. Quote the model's number instead of the routing engine's. |
| Upgrade the top 3 hubs | Rs 16,965 | Rs 2.6 lakh | Removing half the excess delay at those hubs. Range Rs 1.0 lakh-Rs 4.3 lakh annualised. |

**Read these as ranges, not forecasts.** Every input is an assumption we have written down: Rs 450 revenue per leg, a 6%-14% penalty band, and a 30%-65% range for how much excess a hub upgrade removes. We have deliberately not sized this against the full gap versus the routing engine - most of that gap is forecast error, and a bigger dock does not fix a forecast. Sizing the capital case against the whole gap would overstate the return by roughly seven times.

The period sampled is 24 days, so annualised figures assume the same run-rate for a year. September-October includes festive volume; treat the annualisation as indicative until we re-run on a full year.

## What we would like agreed

1. **Now - quote the model, not the engine.** Run it in shadow for four weeks against live shipments, then switch customer-facing promises over. Watch the cold-start fallback rate as the primary health metric.
2. **Quarter 1 - Gurgaon Bilaspur HB (Haryana).** The single largest contributor to excess delay. Capacity and a second dispatch wave.
3. **Quarter 2 - Bangalore Nelmngla H (Karnataka) and Bhiwandi Mankoli HB (Maharashtra).** Completes the top-three case.
4. **In parallel - send us the rate card.** The FTL/Carting framework is built and runs; it is currently priced on our assumptions rather than your numbers. With the real rate card it becomes a dispatch rule rather than a recommendation.
5. **Separately - the 112 artifact corridors** go to data quality, not to operations.

---

*Every figure in this memo is generated directly from the analysis pipeline; changing an assumption in the configuration regenerates the memo with it. Method, model comparison and limitations are in the technical appendix.*
