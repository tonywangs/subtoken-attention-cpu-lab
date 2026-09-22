# Variable-length recall results

Generated from preserved artifacts; accuracy is the unweighted mean across three seeds.

Frozen configuration SHA-256: `3389af5422c846b596a4520377811400ad5441c2d1016d22eeb663f6a75cd709`.
Width 32; 1363 updates per run; 512 sequences / 2,048 queries per seed and length.

| Length | Method | Query accuracy % (seed min–max) | Exact sequence % | Inference ms / batch 32 | Ratio to MLP |
|---:|---|---:|---:|---:|---:|
| 48 | mlp | 100.00 (100.00–100.00) | 100.00 | 20.412 | 1.000 |
| 48 | pool | 100.00 (100.00–100.00) | 100.00 | 20.396 | 0.999 |
| 48 | nonlinear | 100.00 (100.00–100.00) | 100.00 | 21.262 | 1.042 |
| 48 | attention | 100.00 (100.00–100.00) | 100.00 | 26.056 | 1.276 |
| 72 | mlp | 99.82 (99.66–100.00) | 99.28 | 38.450 | 1.000 |
| 72 | pool | 99.71 (99.37–99.95) | 98.83 | 38.029 | 0.989 |
| 72 | nonlinear | 99.90 (99.80–100.00) | 99.61 | 38.992 | 1.014 |
| 72 | attention | 99.59 (98.97–99.95) | 98.37 | 46.397 | 1.207 |
| 96 | mlp | 99.74 (99.61–99.95) | 98.96 | 61.028 | 1.000 |
| 96 | pool | 99.04 (98.24–99.90) | 96.16 | 61.052 | 1.001 |
| 96 | nonlinear | 99.77 (99.56–99.90) | 99.09 | 62.051 | 1.017 |
| 96 | attention | 99.06 (98.29–99.95) | 96.22 | 71.374 | 1.170 |

## Paired query accuracy differences

Percentage points relative to MLP; seed order 111, 222, 333. No significance claim.

| Length | Method | Seed differences (pp) | Mean (pp) |
|---:|---|---|---:|
| 48 | pool | +0.00, +0.00, +0.00 | +0.00 |
| 48 | nonlinear | +0.00, +0.00, +0.00 | +0.00 |
| 48 | attention | +0.00, +0.00, +0.00 | +0.00 |
| 72 | pool | +0.15, -0.05, -0.44 | -0.11 |
| 72 | nonlinear | +0.24, +0.00, +0.00 | +0.08 |
| 72 | attention | +0.29, -0.15, -0.83 | -0.23 |
| 96 | pool | -0.68, -0.05, -1.37 | -0.70 |
| 96 | nonlinear | -0.10, -0.05, +0.24 | +0.03 |
| 96 | attention | -0.73, +0.00, -1.32 | -0.68 |

## Retrieval distance

Query position minus the corresponding definition value position. Counts pool three seeds.

| Length | Method | Distance | Queries | Accuracy % |
|---:|---|---|---:|---:|
| 48 | mlp | 1-16 | 1931 | 100.00 |
| 48 | mlp | 17-32 | 2707 | 100.00 |
| 48 | mlp | 33-64 | 1506 | 100.00 |
| 48 | pool | 1-16 | 1931 | 100.00 |
| 48 | pool | 17-32 | 2707 | 100.00 |
| 48 | pool | 33-64 | 1506 | 100.00 |
| 48 | nonlinear | 1-16 | 1931 | 100.00 |
| 48 | nonlinear | 17-32 | 2707 | 100.00 |
| 48 | nonlinear | 33-64 | 1506 | 100.00 |
| 48 | attention | 1-16 | 1931 | 100.00 |
| 48 | attention | 17-32 | 2707 | 100.00 |
| 48 | attention | 33-64 | 1506 | 100.00 |
| 72 | mlp | 1-16 | 1094 | 100.00 |
| 72 | mlp | 17-32 | 1613 | 100.00 |
| 72 | mlp | 33-64 | 3235 | 99.72 |
| 72 | mlp | 65-128 | 202 | 99.01 |
| 72 | pool | 1-16 | 1094 | 100.00 |
| 72 | pool | 17-32 | 1613 | 100.00 |
| 72 | pool | 33-64 | 3235 | 99.57 |
| 72 | pool | 65-128 | 202 | 98.02 |
| 72 | nonlinear | 1-16 | 1094 | 100.00 |
| 72 | nonlinear | 17-32 | 1613 | 100.00 |
| 72 | nonlinear | 33-64 | 3235 | 99.85 |
| 72 | nonlinear | 65-128 | 202 | 99.50 |
| 72 | attention | 1-16 | 1094 | 100.00 |
| 72 | attention | 17-32 | 1613 | 100.00 |
| 72 | attention | 33-64 | 3235 | 99.32 |
| 72 | attention | 65-128 | 202 | 98.51 |
| 96 | mlp | 1-16 | 813 | 100.00 |
| 96 | mlp | 17-32 | 1140 | 100.00 |
| 96 | mlp | 33-64 | 2381 | 99.79 |
| 96 | mlp | 65-128 | 1810 | 99.39 |
| 96 | pool | 1-16 | 813 | 100.00 |
| 96 | pool | 17-32 | 1140 | 100.00 |
| 96 | pool | 33-64 | 2381 | 99.54 |
| 96 | pool | 65-128 | 1810 | 97.35 |
| 96 | nonlinear | 1-16 | 813 | 100.00 |
| 96 | nonlinear | 17-32 | 1140 | 100.00 |
| 96 | nonlinear | 33-64 | 2381 | 99.79 |
| 96 | nonlinear | 65-128 | 1810 | 99.50 |
| 96 | attention | 1-16 | 813 | 100.00 |
| 96 | attention | 17-32 | 1140 | 100.00 |
| 96 | attention | 33-64 | 2381 | 99.33 |
| 96 | attention | 65-128 | 1810 | 97.68 |

## Training cost

Wall time includes validation; sequential runs on a shared host are not controlled timing comparisons.

| Method | Parameters | Total training seconds (3 seeds) |
|---|---:|---:|
| mlp | 16827 | 158.38 |
| pool | 16745 | 162.53 |
| nonlinear | 16745 | 170.14 |
| attention | 16745 | 225.03 |

Uniform valid-value guessing has expected query accuracy 6.25%; uniform full-vocabulary guessing 2.44%.
These are analytical baselines, not measured trained models. Four independent uniform value guesses give 0.001526% expected exact accuracy.

See [protocol and limitations](MQAR_PROTOCOL.md) and [interpretation](MQAR_INTERPRETATION.md).
