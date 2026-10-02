# DROID 500-Episode Shard: Data Contract & Alignment Report (Milestone E3.1)

- **Dataset Source:** `droid:1.0.1` (gs://gresearch/robotics)
- **Generated UTC:** 2026-10-02T05:56:57.457352+00:00
- **Total Shard Episodes:** 500
- **Split Counts:** Train=400, Val=50, Test=50
- **Total Unique Robot Laboratories (Sites):** 14

## 1. Laboratory & Site Stratification Breakdown

| Site (Lab) | Total Episodes | Train (80%) | Val (10%) | Test (10%) |
|:---|:---:|:---:|:---:|:---:|
| `TRI` | 127 | 104 | 8 | 15 |
| `AUTOLab` | 72 | 63 | 5 | 4 |
| `IRIS` | 45 | 32 | 8 | 5 |
| `RAIL` | 45 | 38 | 4 | 3 |
| `ILIAD` | 44 | 34 | 4 | 6 |
| `IPRL` | 44 | 33 | 5 | 6 |
| `BVL` | 29 | 22 | 4 | 3 |
| `CLVR` | 25 | 21 | 4 | 0 |
| `REAL` | 19 | 17 | 0 | 2 |
| `PennPAL` | 14 | 8 | 3 | 3 |
| `RPL` | 12 | 10 | 1 | 1 |
| `WEIRD` | 12 | 8 | 3 | 1 |
| `GuptaLab` | 7 | 6 | 1 | 0 |
| `RAD` | 5 | 4 | 0 | 1 |

## 2. In-Depth Alignment & Contract Verification (8 Inspected Episodes)

| # | Episode ID | Site | Split | Steps | Wrist Var | Ext Var | Act Norm | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | `98ca4d8e47383c1482a0f17d` | `AUTOLab` | `train` | 265 | 2674.7 | 3867.5 | 3.032 | **PASS_CONTRACT** |
| 2 | `fcfbe2da8ea2d12c55f3dbbf` | `ILIAD` | `train` | 245 | 1274.4 | 923.6 | 3.209 | **PASS_CONTRACT** |
| 3 | `50bf51d770bfb4a9b14d8090` | `IRIS` | `train` | 199 | 5550.6 | 4943.7 | 3.204 | **PASS_CONTRACT** |
| 4 | `100a854030acdde2eaad3734` | `TRI` | `train` | 290 | 6409.3 | 4905.5 | 3.278 | **PASS_CONTRACT** |
| 5 | `0834c3f39ebb9cbe6a31dace` | `RAIL` | `train` | 125 | 4843.6 | 5312.6 | 3.106 | **PASS_CONTRACT** |
| 6 | `020ce061d83d879c0ed120c1` | `TRI` | `train` | 562 | 4542.9 | 4160.3 | 2.986 | **PASS_CONTRACT** |
| 7 | `7be49481735244064e493d28` | `AUTOLab` | `train` | 427 | 2694.4 | 1970.1 | 3.200 | **PASS_CONTRACT** |
| 8 | `543de661de988b9998cd3d1b` | `AUTOLab` | `train` | 286 | 4930.1 | 3961.6 | 3.199 | **PASS_CONTRACT** |

## 3. Inspected Episode Task Instructions

1. **`98ca4d8e47383c1482a0f17d`** (AUTOLab, train): *"Put the black objects into the drawer and close the drawer."*
   - Cartesian Range: X=[0.44, 0.78], Z=[0.13, 0.43]
   - Gripper Range: [0.000, 0.000]
   - Joint Range: [-2.51, 3.18]

2. **`fcfbe2da8ea2d12c55f3dbbf`** (ILIAD, train): *"Pick up the lid and put it on top of the black pot"*
   - Cartesian Range: X=[0.34, 0.54], Z=[0.31, 0.48]
   - Gripper Range: [0.000, 0.612]
   - Joint Range: [-2.74, 2.61]

3. **`50bf51d770bfb4a9b14d8090`** (IRIS, train): *"Remove the pen from the cup and place it on the table"*
   - Cartesian Range: X=[0.40, 0.53], Z=[0.15, 0.45]
   - Gripper Range: [0.000, 0.877]
   - Joint Range: [-2.73, 2.89]

4. **`100a854030acdde2eaad3734`** (TRI, train): *"Pick up the white and green brush from the table and hang it on the wall"*
   - Cartesian Range: X=[0.29, 0.59], Z=[0.33, 0.81]
   - Gripper Range: [0.000, 0.806]
   - Joint Range: [-2.71, 2.81]

5. **`0834c3f39ebb9cbe6a31dace`** (RAIL, train): *"Pick up the red pepper on the plate and put it on the table"*
   - Cartesian Range: X=[0.34, 0.67], Z=[0.10, 0.54]
   - Gripper Range: [0.000, 0.449]
   - Joint Range: [-2.26, 2.22]

6. **`020ce061d83d879c0ed120c1`** (TRI, train): *"Fold the yellow towel twice from left to right."*
   - Cartesian Range: X=[0.28, 0.62], Z=[0.20, 0.56]
   - Gripper Range: [0.000, 0.991]
   - Joint Range: [-2.55, 3.03]

7. **`7be49481735244064e493d28`** (AUTOLab, train): *"Remove the two t-shirts and two towels from the white box and place them on the table"*
   - Cartesian Range: X=[0.41, 0.63], Z=[0.18, 0.68]
   - Gripper Range: [0.000, 0.960]
   - Joint Range: [-2.20, 2.37]

8. **`543de661de988b9998cd3d1b`** (AUTOLab, train): *""*
   - Cartesian Range: X=[0.38, 0.77], Z=[0.26, 0.56]
   - Gripper Range: [0.000, 0.991]
   - Joint Range: [-2.30, 2.54]

