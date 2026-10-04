# Session 6A: Fixed-Budget Spatial Patch Selection

**Mode:** `real` · **Device:** `cuda` · **Run:** `spatial_patches_20261003` · **Patches:** `32` (`16` per camera) · **beta:** `0.5`

## Budget k = 4 (2 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.01078 | 0.00660 | 0.00951 | 0.00875 |
| `stratified_random` | 0.01075 | 0.00645 | 0.00950 | 0.00896 |
| `early_feature_norm` | 0.00713 | 0.00656 | 0.00657 | 0.00418 |
| `early_cls_attention` | 0.00869 | 0.00648 | 0.00764 | 0.00705 |
| `direct_ranking_critic` | 0.01037 | 0.00796 | 0.00938 | 0.00782 |
| `direct_critic_privileged` | 0.01115 | 0.00747 | 0.00970 | 0.00966 |
| `second_order_curvature` | 0.00678 | 0.00520 | 0.00598 | 0.00557 |
| `belief_space_voi` | 0.00678 | 0.00520 | 0.00598 | 0.00557 |
| `exact_costate_reference` | 0.00013 | 0.00000 | 0.00007 | 0.00027 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0000` over `3456` selections
- Submodularity violation rate: `0.1155` (`133` / `1152`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00591`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `24985.5`, p = `3.764e-08` (n = `384`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `22050.0`, p = `1.133e-11` (n = `384`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `384`)

## Budget k = 8 (4 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00796 | 0.00514 | 0.00690 | 0.00648 |
| `stratified_random` | 0.00805 | 0.00535 | 0.00709 | 0.00630 |
| `early_feature_norm` | 0.00702 | 0.00601 | 0.00646 | 0.00442 |
| `early_cls_attention` | 0.00733 | 0.00542 | 0.00633 | 0.00627 |
| `direct_ranking_critic` | 0.00930 | 0.00634 | 0.00820 | 0.00725 |
| `direct_critic_privileged` | 0.00998 | 0.00626 | 0.00865 | 0.00882 |
| `second_order_curvature` | 0.00684 | 0.00543 | 0.00607 | 0.00487 |
| `belief_space_voi` | 0.00684 | 0.00543 | 0.00607 | 0.00487 |
| `exact_costate_reference` | 0.00011 | 0.00005 | 0.00008 | 0.00017 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0001` over `3456` selections
- Submodularity violation rate: `0.2254` (`606` / `2688`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00599`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `27934.5`, p = `3.374e-05` (n = `384`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `21635.0`, p = `1.909e-12` (n = `384`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `384`)

## Budget k = 16 (8 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00662 | 0.00423 | 0.00587 | 0.00518 |
| `stratified_random` | 0.00516 | 0.00357 | 0.00456 | 0.00387 |
| `early_feature_norm` | 0.00488 | 0.00401 | 0.00448 | 0.00296 |
| `early_cls_attention` | 0.00442 | 0.00286 | 0.00372 | 0.00398 |
| `direct_ranking_critic` | 0.00642 | 0.00487 | 0.00574 | 0.00457 |
| `direct_critic_privileged` | 0.00678 | 0.00432 | 0.00588 | 0.00606 |
| `second_order_curvature` | 0.00508 | 0.00415 | 0.00440 | 0.00388 |
| `belief_space_voi` | 0.00508 | 0.00415 | 0.00440 | 0.00388 |
| `exact_costate_reference` | 0.00004 | 0.00001 | 0.00002 | 0.00008 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0001` over `3456` selections
- Submodularity violation rate: `0.3983` (`2294` / `5760`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00438`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `26689.5`, p = `2.374e-06` (n = `384`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `22870.5`, p = `9.588e-11` (n = `384`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `384`)

## Budget k = 4 (2 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.01078 | 0.00660 | 0.00951 | 0.00875 |
| `stratified_random` | 0.01075 | 0.00645 | 0.00950 | 0.00896 |
| `early_feature_norm` | 0.00713 | 0.00656 | 0.00657 | 0.00418 |
| `early_cls_attention` | 0.00869 | 0.00648 | 0.00764 | 0.00705 |
| `direct_ranking_critic` | 0.01139 | 0.00766 | 0.01021 | 0.00888 |
| `direct_critic_privileged` | 0.01173 | 0.00998 | 0.01061 | 0.00869 |
| `second_order_curvature` | 0.00769 | 0.00689 | 0.00714 | 0.00558 |
| `belief_space_voi` | 0.00769 | 0.00689 | 0.00714 | 0.00558 |
| `exact_costate_reference` | 0.00013 | 0.00000 | 0.00007 | 0.00027 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0001` over `3456` selections
- Submodularity violation rate: `0.1155` (`133` / `1152`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00707`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `23066.0`, p = `1.731e-10` (n = `384`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `21856.0`, p = `3.937e-12` (n = `384`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `384`)

## Budget k = 8 (4 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00796 | 0.00514 | 0.00690 | 0.00648 |
| `stratified_random` | 0.00805 | 0.00535 | 0.00709 | 0.00630 |
| `early_feature_norm` | 0.00702 | 0.00601 | 0.00646 | 0.00442 |
| `early_cls_attention` | 0.00733 | 0.00542 | 0.00633 | 0.00627 |
| `direct_ranking_critic` | 0.00999 | 0.00678 | 0.00887 | 0.00776 |
| `direct_critic_privileged` | 0.01083 | 0.00863 | 0.00963 | 0.00837 |
| `second_order_curvature` | 0.00777 | 0.00669 | 0.00695 | 0.00530 |
| `belief_space_voi` | 0.00777 | 0.00669 | 0.00695 | 0.00530 |
| `exact_costate_reference` | 0.00011 | 0.00005 | 0.00008 | 0.00017 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0001` over `3456` selections
- Submodularity violation rate: `0.2254` (`606` / `2688`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00687`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `25471.0`, p = `1.303e-07` (n = `384`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `20541.5`, p = `4.579e-14` (n = `384`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `384`)

## Budget k = 16 (8 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00662 | 0.00423 | 0.00587 | 0.00518 |
| `stratified_random` | 0.00516 | 0.00357 | 0.00456 | 0.00387 |
| `early_feature_norm` | 0.00488 | 0.00401 | 0.00448 | 0.00296 |
| `early_cls_attention` | 0.00442 | 0.00286 | 0.00372 | 0.00398 |
| `direct_ranking_critic` | 0.00708 | 0.00510 | 0.00612 | 0.00559 |
| `direct_critic_privileged` | 0.00729 | 0.00574 | 0.00658 | 0.00545 |
| `second_order_curvature` | 0.00548 | 0.00460 | 0.00489 | 0.00385 |
| `belief_space_voi` | 0.00548 | 0.00460 | 0.00489 | 0.00385 |
| `exact_costate_reference` | 0.00004 | 0.00001 | 0.00002 | 0.00008 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0001` over `3456` selections
- Submodularity violation rate: `0.3983` (`2294` / `5760`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00487`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `22792.0`, p = `7.546e-11` (n = `384`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `17843.5`, p = `1.592e-18` (n = `384`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `384`)

## Budget k = 4 (2 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.01078 | 0.00660 | 0.00951 | 0.00875 |
| `stratified_random` | 0.01075 | 0.00645 | 0.00950 | 0.00896 |
| `early_feature_norm` | 0.00713 | 0.00656 | 0.00657 | 0.00418 |
| `early_cls_attention` | 0.00869 | 0.00648 | 0.00764 | 0.00705 |
| `direct_ranking_critic` | 0.01021 | 0.00673 | 0.00915 | 0.00823 |
| `direct_critic_privileged` | 0.01197 | 0.00854 | 0.01080 | 0.00890 |
| `second_order_curvature` | 0.00775 | 0.00647 | 0.00716 | 0.00574 |
| `belief_space_voi` | 0.00775 | 0.00647 | 0.00716 | 0.00574 |
| `exact_costate_reference` | 0.00013 | 0.00000 | 0.00007 | 0.00027 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0001` over `3456` selections
- Submodularity violation rate: `0.1155` (`133` / `1152`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00708`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `32844.0`, p = `0.05864` (n = `384`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `21542.5`, p = `1.406e-12` (n = `384`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `384`)

## Budget k = 8 (4 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00796 | 0.00514 | 0.00690 | 0.00648 |
| `stratified_random` | 0.00805 | 0.00535 | 0.00709 | 0.00630 |
| `early_feature_norm` | 0.00702 | 0.00601 | 0.00646 | 0.00442 |
| `early_cls_attention` | 0.00733 | 0.00542 | 0.00633 | 0.00627 |
| `direct_ranking_critic` | 0.00897 | 0.00613 | 0.00781 | 0.00735 |
| `direct_critic_privileged` | 0.01036 | 0.00788 | 0.00941 | 0.00745 |
| `second_order_curvature` | 0.00751 | 0.00568 | 0.00651 | 0.00567 |
| `belief_space_voi` | 0.00751 | 0.00568 | 0.00651 | 0.00567 |
| `exact_costate_reference` | 0.00011 | 0.00005 | 0.00008 | 0.00017 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0002` over `3456` selections
- Submodularity violation rate: `0.2254` (`606` / `2688`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00643`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `36492.5`, p = `0.8301` (n = `384`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `23496.0`, p = `6.174e-10` (n = `384`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `384`)

## Budget k = 16 (8 per camera)

| Policy | Mean regret | Median | 10% trimmed mean | Std |
|---|---:|---:|---:|---:|
| `uniform_grid` | 0.00662 | 0.00423 | 0.00587 | 0.00518 |
| `stratified_random` | 0.00516 | 0.00357 | 0.00456 | 0.00387 |
| `early_feature_norm` | 0.00488 | 0.00401 | 0.00448 | 0.00296 |
| `early_cls_attention` | 0.00442 | 0.00286 | 0.00372 | 0.00398 |
| `direct_ranking_critic` | 0.00669 | 0.00418 | 0.00599 | 0.00546 |
| `direct_critic_privileged` | 0.00709 | 0.00516 | 0.00639 | 0.00533 |
| `second_order_curvature` | 0.00545 | 0.00381 | 0.00467 | 0.00440 |
| `belief_space_voi` | 0.00545 | 0.00381 | 0.00467 | 0.00440 |
| `exact_costate_reference` | 0.00004 | 0.00001 | 0.00002 | 0.00008 |
| `calibrated_greedy_oracle` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- Additivity R^2: `0.0002` over `3456` selections
- Submodularity violation rate: `0.3983` (`2294` / `5760`)
- Distillation gap (exact reference − VOI trimmed mean): `-0.00465`
- Paired Wilcoxon `voi_minus_direct_ranking`: statistic `30457.0`, p = `0.002811` (n = `384`)
- Paired Wilcoxon `voi_minus_direct_critic_privileged`: statistic `23628.5`, p = `9.066e-10` (n = `384`)
- Paired Wilcoxon `voi_minus_second_order_curvature`: statistic `0.0`, p = `1` (n = `384`)

## Belief-Space Weight Sensitivity (spec B2)

| beta | Mean regret | 10% trimmed mean |
|---:|---:|---:|
| **k_total=4** | | |
| `beta=+0.0` | 0.00775 | 0.00716 |
| `beta=+0.5` | 0.00775 | 0.00716 |
| `beta=+1.0` | 0.00775 | 0.00716 |
| `beta=-0.5` | 0.00775 | 0.00716 |
| `beta=-1.0` | 0.00775 | 0.00716 |
| **k_total=8** | | |
| `beta=+0.0` | 0.00751 | 0.00651 |
| `beta=+0.5` | 0.00751 | 0.00651 |
| `beta=+1.0` | 0.00751 | 0.00651 |
| `beta=-0.5` | 0.00751 | 0.00651 |
| `beta=-1.0` | 0.00751 | 0.00651 |
| **k_total=16** | | |
| `beta=+0.0` | 0.00545 | 0.00467 |
| `beta=+0.5` | 0.00545 | 0.00467 |
| `beta=+1.0` | 0.00545 | 0.00467 |
| `beta=-0.5` | 0.00545 | 0.00467 |
| `beta=-1.0` | 0.00545 | 0.00467 |

`beta = 0` must equal `second_order_curvature`; negative `beta` is the sign-flip control.

## Greedy Oracle Calibration (spec B5)

- k_cam = 2 over 2 windows: mean excess objective vs exhaustive `1.097e-05`, max `2.098e-05`, optimal on `50.0%` of windows

## Exit Gate (pre-registered, spec section 5)

### k=4

- Belief-space VOI trimmed mean regret: `0.00716`
- Advantage over `uniform_grid`: `24.72%` (threshold `8.0%`)
- Advantage over `early_feature_norm`: `-8.88%` (threshold `8.0%`)
- Criterion 1 (primary advantage): **NOT MET**
- Criterion 2 (non-inferiority vs privileged critic): **MET**
- Wilcoxon p (VOI vs curvature): `1`

### k=8

- Belief-space VOI trimmed mean regret: `0.00651`
- Advantage over `uniform_grid`: `5.59%` (threshold `8.0%`)
- Advantage over `early_feature_norm`: `-0.84%` (threshold `8.0%`)
- Criterion 1 (primary advantage): **NOT MET**
- Criterion 2 (non-inferiority vs privileged critic): **MET**
- Wilcoxon p (VOI vs curvature): `1`

### k=16

- Belief-space VOI trimmed mean regret: `0.00467`
- Advantage over `uniform_grid`: `20.47%` (threshold `8.0%`)
- Advantage over `early_feature_norm`: `-4.15%` (threshold `8.0%`)
- Criterion 1 (primary advantage): **NOT MET**
- Criterion 2 (non-inferiority vs privileged critic): **MET**
- Wilcoxon p (VOI vs curvature): `1`

- Criterion 1 across all budgets: **NOT MET**
- Criterion 2 across all budgets: **MET**

**Branch: `negative_result: geometric coverage dominates downstream sensitivity`**

## Caveats

- Latency is deliberately not claimed here; wall-clock token selection is Session 6C (spec Q4).
- grounded epistemic VOI costs one rollout per patch, unlike the O(1) cotangent scorers.
- A checkpoint not trained with patch dropout is off-distribution at k < P (spec B6).
- Regret is measured against a greedy rollout *reference*, not a proven optimum; it is a lower bound and may be negative where a policy beats greedy.
- Deployable scorers need one encode per patch (latent perturbation) and, for belief-space VOI, one rollout per patch; only the cotangent term is O(1).
