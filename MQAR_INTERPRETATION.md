# Interpretation of the frozen recall experiment

Within-token attention showed no accuracy advantage in this completed comparison
and cost more CPU inference time. All twelve checkpoints achieved 100% query and
exact-sequence accuracy at length 48. At length 96, mean query accuracy was
99.74% for MLP, 99.77% for nonlinear pooling, 99.04% for linear pooling, and 99.06%
for within-token attention. The [generated results](MQAR_RESULTS.md) preserve
individual paired differences, seed ranges, exact-sequence accuracy, distance
breakdowns and measured costs. Three seeds and near-saturated scores do not
establish a reliable general architecture ranking.

Attention's length-96 differences against the paired MLP were -0.73, 0.00 and
-1.32 percentage points. The nonlinear control differed from MLP by -0.10, -0.05
and +0.24 points. There is no evidence here that interactions among learned
projections improve recall over an ordinary MLP or independent GELU projections
followed by pooling. This is an empirical finding for the frozen setup, not a
proof that the mechanism cannot help elsewhere.

Errors concentrate at longer retrieval distances. At length 96 and distances
65–128, the 1,810 pooled queries had accuracy 99.39% for MLP, 99.50% for nonlinear
pooling, 97.35% for linear pooling and 97.68% for attention. All methods had
100% accuracy for distances up to 32 in every condition. Longer sequences change
both the distribution of positions and distractor count; these results cannot
separate position extrapolation from retrieval-distance sensitivity.

The balanced inference pass measured attention/MLP median-time ratios, averaged
across paired seeds, of 1.276 at length 48, 1.207 at length 72 and 1.170 at length
96. Nonlinear pooling ratios were 1.042, 1.014 and 1.017. The attention module's
extra operations did not earn an accuracy gain here. These are batch-32 CPU
measurements of the implemented operators; common quadratic sequence attention
can make per-token overhead a smaller fraction at longer lengths. That last
explanation is an interpretation, not a separate profiling measurement.

Training wall time totaled 716.08 seconds across twelve runs, including
validation. Each run used 1,363 updates at width 32. Total parameters are 16,827
for MLP and 16,745 for each other method, a maximum/minimum difference below 0.5%.
Shared parameters and projection-control parameters started identically within
seeds, and training sample/order hashes match. All models retain nonlinear
ordinary sequence attention and LayerNorm; the linear-pooling result is **not**
evidence that a fully linear sequence model solves associative recall.

## What was frozen and preserved

The protocol, source hashes, pilot and configuration were frozen before the
first final run. The timing-only development pilot covered two widths, four
architectures and two separate seeds, with 100 updates per condition. It chose
width 32 and 1,363 steps by the preregistered arithmetic. One development timing
segment (width 32, seed 99002, MLP) overlapped a separate parameter-count audit
process launched by this task; its median was about 88 ms, compared with about
42 ms for the other width-32 MLP segment. Those measured samples were retained
without rerunning or adjusting the selection. This made the selected training
budget more conservative. Host variation may also have contributed; the overlap
was not a controlled causal measurement.

No protocol, model, training setting or checkpoint-selection rule changed after
final evaluation began. There was no early stopping or best-validation selection.
The final inference timing pass ran after all training and before package replay,
with no concurrent compute check launched by this task. Other host tenants were
uncontrolled. Verification re-evaluates the same fixed checkpoints; it is not
additional tuning.

`results/mqar` contains the exact input shards, labels, retrieval distances,
lengths, predictions, twelve checkpoints, every training loss and gradient norm,
validation curves, environment, initial-weight and order hashes, raw balanced
timing samples, and an artifact manifest. `configs/mqar-frozen.json` and
`MQAR_PROTOCOL.md` retain the preregistered rules. See `results/dependencies.txt`
for the shared installed dependency versions; the new run also records Python,
NumPy, PyTorch, CPU identity, thread counts and PyTorch build details in its own
environment artifact. The earlier experiment's sources, configurations and
results remain unchanged.

The replay CLI verifies artifacts, regenerates all data, checks the dictionary
oracle and split disjointness, checks pairing, and reproduces all 36 held-out
checkpoint evaluations. Tests additionally cover causal and padding behavior,
variable-length batching, shared initialization and nonlinear forward/gradient
reference checks. The offline installation check builds a wheel, installs it in
a temporary environment and runs the installed CLI outside the checkout. It
reuses local dependency files through symlinks, excludes editable-project and
`.pth` entries, and does not claim to recreate dependencies from a fresh download.

## Limits and possible future experiments

The task remains easy for these small sequence-attention models. A six-pair,
16-key vocabulary with a disjoint distractor alphabet is a CPU-scale MQAR variant,
not published Zoology MQAR. Length 48 saturation limits discrimination; even
length 96 performance is very high. Three independent initialization/data seeds
provide descriptive paired comparisons, not strong statistical evidence.
Equal updates and nearly equal parameter counts do not equalize effective
capacity, optimization difficulty or wall time. Expanded linear pooling is not
an optimized fused affine baseline. CPU timing cannot predict GPU performance,
and this synthetic task cannot establish language-model usefulness.

A separate future protocol could increase association load and vocabulary or
make distractors less distinguishable. It would need new development and final
seeds, a separately frozen budget and fresh test inputs. The completed results
here must remain intact rather than becoming a tuning set for revised claims.
