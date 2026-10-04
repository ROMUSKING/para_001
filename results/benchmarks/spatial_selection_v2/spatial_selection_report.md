# Session 6A: Fixed-Budget Spatial Patch Selection

**Mode:** `real` · **Device:** `cuda` · **Run:** `spatial_patches_20261003` · **Patches:** `32` (`16` per camera) · **beta:** `0.5`

## Budget k = 4 (2 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00528 | 0.00437 | 0.00485 | 0.00344 |
| `stratified_random` | 0.00528 | 0.00442 | 0.00476 | 0.00372 |
| `early_feature_norm` | 0.00505 | 0.00408 | 0.00443 | 0.00379 |
| `early_cls_attention` | 0.00517 | 0.00443 | 0.00464 | 0.00385 |
| `direct_ranking_critic` | 0.00532 | 0.00429 | 0.00464 | 0.00416 |
| `direct_critic_privileged` | 0.00552 | 0.00453 | 0.00502 | 0.00404 |
| `second_order_curvature` | 0.00528 | 0.00434 | 0.00464 | 0.00446 |
| `belief_space_voi` | 0.00528 | 0.00434 | 0.00464 | 0.00446 |
| `exact_costate_reference` | 0.00013 | 0.00002 | 0.00008 | 0.00024 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0000` over `10800` selections
- Submodularity violation rate: `0.1267` (`456` / `3600`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00456`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `358223.0`, p = `0.9416` (n = `1200`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `326490.5`, p = `0.006478` (n = `1200`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `1200`)

## Budget k = 8 (4 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00446 | 0.00380 | 0.00412 | 0.00253 |
| `stratified_random` | 0.00454 | 0.00394 | 0.00418 | 0.00266 |
| `early_feature_norm` | 0.00457 | 0.00380 | 0.00409 | 0.00311 |
| `early_cls_attention` | 0.00457 | 0.00387 | 0.00403 | 0.00345 |
| `direct_ranking_critic` | 0.00491 | 0.00396 | 0.00431 | 0.00377 |
| `direct_critic_privileged` | 0.00495 | 0.00406 | 0.00449 | 0.00362 |
| `second_order_curvature` | 0.00483 | 0.00388 | 0.00424 | 0.00381 |
| `belief_space_voi` | 0.00483 | 0.00388 | 0.00424 | 0.00381 |
| `exact_costate_reference` | 0.00008 | 0.00003 | 0.00006 | 0.00016 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0000` over `10800` selections
- Submodularity violation rate: `0.2470` (`2075` / `8400`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00419`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `350547.0`, p = `0.4167` (n = `1200`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `332548.5`, p = `0.02083` (n = `1200`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `1200`)

## Budget k = 16 (8 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00320 | 0.00268 | 0.00298 | 0.00186 |
| `stratified_random` | 0.00312 | 0.00274 | 0.00288 | 0.00169 |
| `early_feature_norm` | 0.00316 | 0.00266 | 0.00286 | 0.00200 |
| `early_cls_attention` | 0.00307 | 0.00253 | 0.00274 | 0.00209 |
| `direct_ranking_critic` | 0.00338 | 0.00274 | 0.00302 | 0.00238 |
| `direct_critic_privileged` | 0.00345 | 0.00272 | 0.00311 | 0.00255 |
| `second_order_curvature` | 0.00323 | 0.00245 | 0.00282 | 0.00254 |
| `belief_space_voi` | 0.00323 | 0.00245 | 0.00282 | 0.00254 |
| `exact_costate_reference` | 0.00002 | 0.00001 | 0.00002 | 0.00007 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0000` over `10800` selections
- Submodularity violation rate: `0.3827` (`6888` / `18000`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00280`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `322479.0`, p = `0.001634` (n = `1200`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `309955.0`, p = `2.756e-05` (n = `1200`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `1200`)

## Budget k = 4 (2 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00528 | 0.00437 | 0.00485 | 0.00344 |
| `stratified_random` | 0.00528 | 0.00442 | 0.00476 | 0.00372 |
| `early_feature_norm` | 0.00505 | 0.00408 | 0.00443 | 0.00379 |
| `early_cls_attention` | 0.00517 | 0.00443 | 0.00464 | 0.00385 |
| `direct_ranking_critic` | 0.00537 | 0.00442 | 0.00485 | 0.00379 |
| `direct_critic_privileged` | 0.00501 | 0.00431 | 0.00460 | 0.00336 |
| `second_order_curvature` | 0.00520 | 0.00391 | 0.00450 | 0.00466 |
| `belief_space_voi` | 0.00520 | 0.00391 | 0.00450 | 0.00466 |
| `exact_costate_reference` | 0.00013 | 0.00002 | 0.00008 | 0.00024 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0001` over `10800` selections
- Submodularity violation rate: `0.1267` (`456` / `3600`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00442`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `297923.0`, p = `2.587e-07` (n = `1200`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `357289.0`, p = `0.802` (n = `1200`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `1200`)

## Budget k = 8 (4 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00446 | 0.00380 | 0.00412 | 0.00253 |
| `stratified_random` | 0.00454 | 0.00394 | 0.00418 | 0.00266 |
| `early_feature_norm` | 0.00457 | 0.00380 | 0.00409 | 0.00311 |
| `early_cls_attention` | 0.00457 | 0.00387 | 0.00403 | 0.00345 |
| `direct_ranking_critic` | 0.00482 | 0.00401 | 0.00432 | 0.00346 |
| `direct_critic_privileged` | 0.00457 | 0.00390 | 0.00422 | 0.00287 |
| `second_order_curvature` | 0.00483 | 0.00373 | 0.00423 | 0.00391 |
| `belief_space_voi` | 0.00483 | 0.00373 | 0.00423 | 0.00391 |
| `exact_costate_reference` | 0.00008 | 0.00003 | 0.00006 | 0.00016 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0002` over `10800` selections
- Submodularity violation rate: `0.2470` (`2075` / `8400`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00417`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `338404.0`, p = `0.06823` (n = `1200`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `351953.5`, p = `0.5183` (n = `1200`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `1200`)

## Budget k = 16 (8 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00320 | 0.00268 | 0.00298 | 0.00186 |
| `stratified_random` | 0.00312 | 0.00274 | 0.00288 | 0.00169 |
| `early_feature_norm` | 0.00316 | 0.00266 | 0.00286 | 0.00200 |
| `early_cls_attention` | 0.00307 | 0.00253 | 0.00274 | 0.00209 |
| `direct_ranking_critic` | 0.00342 | 0.00267 | 0.00301 | 0.00254 |
| `direct_critic_privileged` | 0.00314 | 0.00266 | 0.00289 | 0.00190 |
| `second_order_curvature` | 0.00325 | 0.00256 | 0.00287 | 0.00251 |
| `belief_space_voi` | 0.00325 | 0.00256 | 0.00287 | 0.00251 |
| `exact_costate_reference` | 0.00002 | 0.00001 | 0.00002 | 0.00007 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0002` over `10800` selections
- Submodularity violation rate: `0.3827` (`6888` / `18000`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00285`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `286548.5`, p = `8.145e-10` (n = `1200`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `360043.0`, p = `0.983` (n = `1200`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `1200`)

## Budget k = 4 (2 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00528 | 0.00437 | 0.00485 | 0.00344 |
| `stratified_random` | 0.00528 | 0.00442 | 0.00476 | 0.00372 |
| `early_feature_norm` | 0.00505 | 0.00408 | 0.00443 | 0.00379 |
| `early_cls_attention` | 0.00517 | 0.00443 | 0.00464 | 0.00385 |
| `direct_ranking_critic` | 0.00536 | 0.00432 | 0.00477 | 0.00389 |
| `direct_critic_privileged` | 0.00489 | 0.00427 | 0.00450 | 0.00341 |
| `second_order_curvature` | 0.00536 | 0.00394 | 0.00466 | 0.00481 |
| `belief_space_voi` | 0.00536 | 0.00394 | 0.00466 | 0.00481 |
| `exact_costate_reference` | 0.00013 | 0.00002 | 0.00008 | 0.00024 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0004` over `10800` selections
- Submodularity violation rate: `0.1267` (`456` / `3600`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00457`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `341149.0`, p = `0.1219` (n = `1200`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `339270.0`, p = `0.07988` (n = `1200`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `1200`)

## Budget k = 8 (4 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00446 | 0.00380 | 0.00412 | 0.00253 |
| `stratified_random` | 0.00454 | 0.00394 | 0.00418 | 0.00266 |
| `early_feature_norm` | 0.00457 | 0.00380 | 0.00409 | 0.00311 |
| `early_cls_attention` | 0.00457 | 0.00387 | 0.00403 | 0.00345 |
| `direct_ranking_critic` | 0.00485 | 0.00396 | 0.00433 | 0.00345 |
| `direct_critic_privileged` | 0.00441 | 0.00393 | 0.00405 | 0.00291 |
| `second_order_curvature` | 0.00482 | 0.00362 | 0.00421 | 0.00410 |
| `belief_space_voi` | 0.00482 | 0.00362 | 0.00421 | 0.00410 |
| `exact_costate_reference` | 0.00008 | 0.00003 | 0.00006 | 0.00016 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0005` over `10800` selections
- Submodularity violation rate: `0.2470` (`2075` / `8400`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00415`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `342847.5`, p = `0.1461` (n = `1200`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `337226.5`, p = `0.05466` (n = `1200`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `1200`)

## Budget k = 16 (8 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00320 | 0.00268 | 0.00298 | 0.00186 |
| `stratified_random` | 0.00312 | 0.00274 | 0.00288 | 0.00169 |
| `early_feature_norm` | 0.00316 | 0.00266 | 0.00286 | 0.00200 |
| `early_cls_attention` | 0.00307 | 0.00253 | 0.00274 | 0.00209 |
| `direct_ranking_critic` | 0.00326 | 0.00274 | 0.00295 | 0.00216 |
| `direct_critic_privileged` | 0.00310 | 0.00269 | 0.00287 | 0.00189 |
| `second_order_curvature` | 0.00334 | 0.00260 | 0.00295 | 0.00266 |
| `belief_space_voi` | 0.00334 | 0.00260 | 0.00295 | 0.00266 |
| `exact_costate_reference` | 0.00002 | 0.00001 | 0.00002 | 0.00007 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0005` over `10800` selections
- Submodularity violation rate: `0.3827` (`6888` / `18000`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00293`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `344005.0`, p = `0.1748` (n = `1200`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `336519.5`, p = `0.04766` (n = `1200`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `1200`)

## Belief-Space Weight Sensitivity (spec B2)

| beta | Mean regret | 10% trimmed mean |
|---:|---:|---:|
| **k_total=4** | | |
| `beta=+0.0` | 0.00536 | 0.00466 |
| `beta=+0.5` | 0.00536 | 0.00466 |
| `beta=+1.0` | 0.00536 | 0.00466 |
| `beta=-0.5` | 0.00536 | 0.00466 |
| `beta=-1.0` | 0.00536 | 0.00466 |
| **k_total=8** | | |
| `beta=+0.0` | 0.00482 | 0.00421 |
| `beta=+0.5` | 0.00482 | 0.00421 |
| `beta=+1.0` | 0.00482 | 0.00421 |
| `beta=-0.5` | 0.00482 | 0.00421 |
| `beta=-1.0` | 0.00482 | 0.00421 |
| **k_total=16** | | |
| `beta=+0.0` | 0.00334 | 0.00295 |
| `beta=+0.5` | 0.00334 | 0.00295 |
| `beta=+1.0` | 0.00334 | 0.00295 |
| `beta=-0.5` | 0.00334 | 0.00295 |
| `beta=-1.0` | 0.00334 | 0.00295 |

`beta = 0` must equal `second_order_curvature`; negative `beta` is the sign-flip control.

## Greedy Oracle Calibration (spec B5)

- k_cam = 2 over 2 windows: mean excess objective vs exhaustive `1.097e-05`, max `2.098e-05`, optimal on `50.0%` of windows

## Exit Gate (pre-registered, spec section 5)

### k=4

- Belief-space VOI trimmed mean regret: `0.00466`
- Advantage over `uniform_grid`: `3.96%` (threshold `8.0%`)
- Advantage over `early_feature_norm`: `-5.18%` (threshold `8.0%`)
- Criterion 1 (primary advantage): **NOT MET**
- Criterion 2 (non-inferiority vs privileged critic): **NOT MET**
- Wilcoxon p (VOI vs curvature): `1`

### k=8

- Belief-space VOI trimmed mean regret: `0.00421`
- Advantage over `uniform_grid`: `-2.21%` (threshold `8.0%`)
- Advantage over `early_feature_norm`: `-2.86%` (threshold `8.0%`)
- Criterion 1 (primary advantage): **NOT MET**
- Criterion 2 (non-inferiority vs privileged critic): **NOT MET**
- Wilcoxon p (VOI vs curvature): `1`

### k=16

- Belief-space VOI trimmed mean regret: `0.00295`
- Advantage over `uniform_grid`: `1.04%` (threshold `8.0%`)
- Advantage over `early_feature_norm`: `-3.24%` (threshold `8.0%`)
- Criterion 1 (primary advantage): **NOT MET**
- Criterion 2 (non-inferiority vs privileged critic): **MET**
- Wilcoxon p (VOI vs curvature): `1`

- Criterion 1 across all budgets: **NOT MET**
- Criterion 2 across all budgets: **NOT MET**

**Branch: `not_evaluated: degenerate beta sweep, so belief-space VOI and second-order curvature are untested rather than equal`**

## Caveats

- Latency is deliberately not claimed here; wall-clock token selection is Session 6C (spec Q4).
- grounded epistemic VOI costs one rollout per patch, unlike the O(1) cotangent scorers.
- A checkpoint not trained with patch dropout is off-distribution at k < P (spec B6).
- Regret is measured against a greedy rollout *reference*, not a proven optimum; it is a lower bound and may be negative where a policy beats greedy.
- Deployable scorers need one encode per patch (latent perturbation) and, for belief-space VOI, one rollout per patch; only the cotangent term is O(1).
