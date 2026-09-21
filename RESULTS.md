# Recorded results

All 18 runs used 1000 optimizer steps, batch 32, and the frozen configuration.
CPU: DO-Regular; PyTorch 2.6.0+cpu; 1 compute thread.
No configuration changes or checkpoint selection used final test results.

Parameters: MLP 15,909; pool and attention 15,827. Maximum difference: 0.518%.

## Each seed

Accuracy and exact match are percentages. TF = teacher forced; AR = autoregressive.
Recall has one target, so token accuracy equals exact match. Timing is batch-32 median milliseconds.

| Task | Seed | Method | Test CE | TF accuracy | TF exact | AR accuracy | AR exact | Inference ms | Train s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| recall | 11 | mlp | 0.00092 | 100.00 | 100.00 | — | — | 3.586 | 26.76 |
| recall | 11 | pool | 0.00102 | 100.00 | 100.00 | — | — | 3.778 | 17.41 |
| recall | 11 | attention | 0.00101 | 100.00 | 100.00 | — | — | 5.143 | 22.45 |
| recall | 22 | mlp | 0.00753 | 100.00 | 100.00 | — | — | 3.614 | 18.62 |
| recall | 22 | pool | 0.00133 | 100.00 | 100.00 | — | — | 3.873 | 20.86 |
| recall | 22 | attention | 0.00125 | 100.00 | 100.00 | — | — | 5.558 | 48.89 |
| recall | 33 | mlp | 0.00130 | 100.00 | 100.00 | — | — | 3.624 | 38.03 |
| recall | 33 | pool | 0.00101 | 100.00 | 100.00 | — | — | 2.381 | 19.22 |
| recall | 33 | attention | 0.00102 | 100.00 | 100.00 | — | — | 5.636 | 21.12 |
| copy | 11 | mlp | 0.00033 | 100.00 | 100.00 | 100.00 | 100.00 | 3.834 | 19.40 |
| copy | 11 | pool | 0.00035 | 100.00 | 100.00 | 100.00 | 100.00 | 3.283 | 18.50 |
| copy | 11 | attention | 0.00034 | 100.00 | 100.00 | 100.00 | 100.00 | 5.885 | 23.17 |
| copy | 22 | mlp | 0.00036 | 100.00 | 100.00 | 100.00 | 100.00 | 3.394 | 20.38 |
| copy | 22 | pool | 0.00036 | 100.00 | 100.00 | 100.00 | 100.00 | 4.248 | 19.42 |
| copy | 22 | attention | 0.00036 | 100.00 | 100.00 | 100.00 | 100.00 | 6.451 | 24.55 |
| copy | 33 | mlp | 0.00034 | 100.00 | 100.00 | 100.00 | 100.00 | 4.344 | 19.35 |
| copy | 33 | pool | 0.00036 | 100.00 | 100.00 | 100.00 | 100.00 | 4.122 | 17.92 |
| copy | 33 | attention | 0.00036 | 100.00 | 100.00 | 100.00 | 100.00 | 5.747 | 22.14 |

Every final run reached 100% held-out target accuracy and exact match, including
autoregressive copy. These tasks saturated: this suite finds no final-accuracy benefit
from within-token attention. Loss and timing still differ, but ceiling accuracy limits
the experiment’s ability to distinguish representation quality.

## Across seeds

Mean ± sample standard deviation over three seeds; not confidence intervals.

| Task | Method | Test CE | TF accuracy % | TF exact % | AR exact % | Inference ms |
|---|---|---:|---:|---:|---:|---:|
| recall | mlp | 0.00325 ± 0.00371 | 100.00 ± 0.00 | 100.00 ± 0.00 | — | 3.61 ± 0.02 |
| recall | pool | 0.00112 ± 0.00018 | 100.00 ± 0.00 | 100.00 ± 0.00 | — | 3.34 ± 0.84 |
| recall | attention | 0.00109 ± 0.00013 | 100.00 ± 0.00 | 100.00 ± 0.00 | — | 5.45 ± 0.27 |
| copy | mlp | 0.00034 ± 0.00001 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.86 ± 0.48 |
| copy | pool | 0.00035 ± 0.00001 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.88 ± 0.52 |
| copy | attention | 0.00035 ± 0.00001 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 6.03 ± 0.37 |

## Paired differences and costs

Each ratio divides the method by the MLP at the same task/seed. Values are mean ± sample SD.
Accuracy difference is in percentage points; ratios greater than one mean slower.

| Task | Method | TF accuracy difference | Inference ratio | Training ratio |
|---|---|---:|---:|---:|
| recall | pool | 0.00 ± 0.00 | 0.93 ± 0.23 | 0.76 ± 0.32 |
| recall | attention | 0.00 ± 0.00 | 1.51 ± 0.07 | 1.34 ± 1.12 |
| copy | pool | 0.00 ± 0.00 | 1.02 ± 0.21 | 0.94 ± 0.02 |
| copy | attention | 0.00 ± 0.00 | 1.59 ± 0.29 | 1.18 ± 0.03 |

## Separate post-training timing pass

Thirty randomized rounds over all 18 checkpoints, after five warmups per checkpoint.
No package installation or training was running in this task during this pass.
Shared-host noise remains possible. Batch size is 32; values summarize three seed medians.

| Task | Method | Inference ms (mean ± SD) | Paired ratio to MLP (mean ± SD) |
|---|---|---:|---:|
| recall | mlp | 3.21 ± 0.07 | 1.00 ± 0.00 |
| recall | pool | 3.27 ± 0.10 | 1.02 ± 0.04 |
| recall | attention | 4.64 ± 0.13 | 1.45 ± 0.06 |
| copy | mlp | 3.70 ± 0.10 | 1.00 ± 0.00 |
| copy | pool | 3.69 ± 0.07 | 1.00 ± 0.04 |
| copy | attention | 5.38 ± 0.10 | 1.45 ± 0.05 |

## Reading these results

The tables preserve every final run; no runs were discarded. High copy accuracy at this
single fixed length does not demonstrate length generalization. Ceiling scores leave open
how these methods behave on harder inputs. Three seeds support descriptive comparisons only.
Pooling is an affine, low-rank ablation; its differences
cannot be attributed solely to the absence of attention. No claim of general improvement is made.

Raw losses, gradient norms, validation curves, held-out predictions, input shards, checkpoints,
hashes, and twenty inference timing samples per run are in `results/final/`.
See [PROTOCOL.md](PROTOCOL.md) for generation rules, architecture, prior work and limitations.
This document is reproduced by `python3 scripts/report.py --check`.
