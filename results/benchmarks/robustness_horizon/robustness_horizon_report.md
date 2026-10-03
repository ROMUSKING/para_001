# Session 5: Robustness, Cost-Model Sensitivity & Multi-Horizon Report

**Date:** 2026-10-03  
**Device:** cuda  
**Num Seeds:** 3  
**Test Windows Evaluated:** 12462  

## 1. Cost-Model Sensitivity Sweep (H=4)

| Cost Regime | Baseline Critic Regret | Belief-Space VOI Regret | VOI Advantage (%) |
|---|---|---|---|
| `zero` | 0.13008 | **0.09469** | **+27.21%** |
| `uniform` | 0.12953 | **0.09710** | **+25.04%** |
| `default` | 0.13129 | **0.10899** | **+16.98%** |
| `high_penalty` | 0.16289 | **0.15915** | **+2.30%** |
| `latency_weighted` | 0.13077 | **0.09975** | **+23.72%** |

## 2. Multi-Horizon Generalization (Default Cost)

| Horizon H | Baseline Critic Regret | Belief-Space VOI Regret | VOI Advantage (%) |
|---|---|---|---|
| H=2 | 0.37324 | **0.16404** | **+56.05%** |
| H=4 | 0.13129 | **0.10899** | **+16.98%** |

## 3. Leave-One-Site-Out (LOSO) Regret Deltas (Default Cost, H=4)

| Dropped Site | Remaining Critic Regret | Remaining VOI Regret | VOI Advantage (%) |
|---|---|---|---|
| `AUTOLab` | 0.15938 | 0.09917 | +37.78% |
| `BVL` | 0.15050 | 0.09547 | +36.56% |
| `ILIAD` | 0.13241 | 0.07317 | +44.74% |
| `IPRL` | 0.14884 | 0.09071 | +39.06% |
| `IRIS` | 0.10752 | 0.09695 | +9.83% |
| `PennPAL` | 0.15004 | 0.09469 | +36.89% |
| `RAD` | 0.14539 | 0.09114 | +37.31% |
| `RAIL` | 0.14827 | 0.09395 | +36.64% |
| `REAL` | 0.14878 | 0.09407 | +36.77% |
| `RPL` | 0.14997 | 0.09563 | +36.23% |
| `TRI` | 0.16309 | 0.09833 | +39.71% |
| `WEIRD` | 0.14828 | 0.09441 | +36.33% |
