# Dynamics parity `dynamics_parity_20260930T214043Z`

Status: **OK**. Anchors within tolerance: True.

| Checkpoint | Mode | Split | BF16 | Windows | Model RMSE | Persistence RMSE | Relative improvement | Passes 2 % |
|---|---|---|---|---:|---:|---:|---:|---|
| v1 | base | train | True | 5910 | 0.1497 | 0.2101 | +0.288 | True |
| v1 | base | train | False | 5910 | 0.1497 | 0.2101 | +0.288 | True |
| v1 | base | validation | True | 796 | 0.2325 | 0.1886 | -0.233 | False |
| v1 | base | validation | False | 796 | 0.2325 | 0.1886 | -0.233 | False |
| v1 | full | train | True | 5910 | 0.1429 | 0.2101 | +0.320 | True |
| v1 | full | train | False | 5910 | 0.1428 | 0.2101 | +0.320 | True |
| v1 | full | validation | True | 796 | 0.2298 | 0.1886 | -0.219 | False |
| v1 | full | validation | False | 796 | 0.2298 | 0.1886 | -0.219 | False |
| v2 | base | train | True | 5910 | 0.1542 | 0.2101 | +0.266 | True |
| v2 | base | train | False | 5910 | 0.1542 | 0.2101 | +0.266 | True |
| v2 | base | validation | True | 796 | 0.2125 | 0.1886 | -0.127 | False |
| v2 | base | validation | False | 796 | 0.2124 | 0.1886 | -0.126 | False |
| v2 | full | train | True | 5910 | 0.1524 | 0.2101 | +0.275 | True |
| v2 | full | train | False | 5910 | 0.1523 | 0.2101 | +0.275 | True |
| v2 | full | validation | True | 796 | 0.2122 | 0.1886 | -0.125 | False |
| v2 | full | validation | False | 796 | 0.2121 | 0.1886 | -0.125 | False |

- The pilot checkpoint also fails the margin on validation in full mode: the failure is a property of the validation split and this metric, not of the v2 training run.

Diagnostic of dynamics against persistence on train and validation windows for two checkpoints (one pilot v2 seed). Not evidence for or against H2; the test split was not read.
