# Frozen variable-length multi-query recall comparison

This is a new experiment. The original `PROTOCOL.md`, `RESULTS.md`, sources,
configuration and `results/final` remain unchanged. No novelty claim is made.
The purpose is to distinguish within-token attention from both linear pooling
and nonlinear projection/pooling under a harder, variable-length task.

## Task and generation

Vocabulary has 41 tokens: pad 0; keys 1–16; values 17–32; distractors 33–40.
Every sequence begins with six adjacent key/value definitions. Keys are uniformly
sampled without replacement; values are independently sampled with replacement.
Four distinct defined keys are sampled and placed at uniformly sampled distinct
positions in the remaining suffix. All other suffix positions contain uniformly
sampled distractors. Predict the associated value **at each query key position**;
all other labels, including padding, are -100 and ignored. No answers are fed
back, and queries never redefine a key. All four queries require exact matching
for sequence accuracy. Retrieval distance is query position minus definition
value position. Distractors have a separate vocabulary, making their exclusion
relatively easy; this is intentionally a modest CPU diagnostic.

Training lengths are uniformly chosen from 32, 40 and 48. Right-pad each stored
training/validation shard to 48; trim each batch to its longest member. Evaluate
512 sequences per seed at each of lengths **48, 72 and 96**. Longer conditions
keep six associations and four queries: they vary distractor count, query
positions and distance, not association load. Their positions beyond 47 are
unseen in training. Use 8,192 training and 256 diagnostic validation sequences.

Generator uses NumPy default_rng/PCG64. Model seeds are 111, 222, 333; data seeds
are model_seed + 10000 + 100000 * split_index in order train, validation, test48,
test72, test96. Reject repeated unordered key/value assignments within or across
these splits, even if queries and distractors differ. Different seed replicates
are not explicitly deduplicated against one another. Generation order is fixed;
train/validation creation does not depend on whether tests are requested.
A separate left-to-right dictionary parser independently checks labels and
retrieval distances from tokens. It does not read generator metadata.

Shards preserve tokens, labels, distances and lengths. Digests hash named arrays,
shape strings and little-endian int64 bytes in that order. All four architectures
use the same shards. For each model seed, a separate NumPy generator with seed
model_seed + 20000 samples batches of 32 with replacement. Save a SHA-256 of all
sampled row-index bytes; verify equality and regenerate the order on replay.

## Models and fairness

Two pre-normalized transformer blocks, four causal sequence-attention heads,
float32, no dropout, ordinary PyTorch MultiheadAttention with both causal and
key-padding masks. Fixed sinusoidal positions (sine/cosine interleaved, base
10000) are added to unscaled token embeddings; no learned position table.
Final LayerNorm and untied 41-class linear readout. All affine layers have biases;
LayerNorm epsilon 1e-5. Pad embedding is fixed to zero, padded labels ignored.

Width d is chosen by the timing pilot from 32 or 48; r=d/2 and K=4. The shared
sequence-attention, embedding, normalization and output weights are constructed
before the per-token modules. Thus each paired seed starts from **identical shared
weights**. Linear pooling, nonlinear pooling and attention also have identical
initial projection weights. The architectures differ only in the residual
per-token sublayer after each sequence-attention residual:

* `attention`: z=Linear(d,4*r)(x), reshaped into four vectors;
  u=mean(softmax(z z^T / sqrt(r)) z), output=Linear(r,d)(u).
  This reuses the original tied-QKV implementation.
* `pool`: same maps, u=mean(z). It is an affine, rank-at-most-r bottleneck.
* `nonlinear`: same maps, u=mean(GELU(z)), with exact GELU. This adds nonlinearity
  before pooling but no interactions among projections before averaging.
* `mlp`: Linear(d,5*d/4+1), exact GELU, Linear(5*d/4+1,d).

All parameters are functional; require max/min total count <1.02. Equal counts
are not equal effective capacity or equal FLOPs. Pooling is expanded rather than
algebraically fused. Nonlinear pooling and attention differ in activation family,
input-dependent mixing and effective capacity, not attention alone. This is a
controlled architectural comparison, not a definitive causal isolation of an
abstract “attention mechanism.”

## Pilot, freeze and training

Development seeds **99001 and 99002** are separate from final seeds. At each width
and architecture run 100 updates, save all losses and final validation metrics,
and time updates 11–100. Never generate development test data. Accuracy is
**not a selection criterion**. For each width take the slowest median update
time across methods and seeds. Select d=48 if 12*2000*slowest_median*1.25 <=1800
seconds, otherwise d=32. Select steps=max(100,min(2000,
floor(1800/(12*slowest_median(selected_d)*1.25)))). This estimates at most 30
minutes, subject to a 100-step floor, not a wall-clock guarantee. It is a bounded
CPU feasibility pilot, not a search for a favorable result.

Before any final training/test evaluation, save the selected configuration,
UTC freeze timestamp, SHA-256 of this protocol, pilot, and all imported experiment
source files. The run rejects changed sources or protocol. Preserve copies of
config, protocol and pilot in the results, plus start/completion UTC timestamps.
The manifest is an integrity record, not an externally witnessed preregistration
or cryptographic proof of chronology.

AdamW learning rate .003, weight decay .01, defaults betas (.9,.999), eps 1e-8;
global gradient norm clipping at 1; deterministic CPU algorithms; one intra-op
and one inter-op thread. Same updates for all methods; no scheduler, early
stopping or tuning. Evaluate validation at step zero, every 200 steps and the
last step, for diagnostics only. The **last update** is the sole checkpoint.
Save loss and pre-clipping gradient norm every update. Training wall time includes
diagnostic evaluation. Rotate model training order across seeds to reduce fixed
order effects, though it cannot fully balance four positions across three seeds.

Do not retune after final test results, including chance-level learning or
saturation. A failure to learn is a valid completed outcome for this budget.

## Metrics and CPU measurement

For each condition and seed report cross-entropy across 41 logits, query accuracy,
exact-sequence accuracy, and accuracy in distance bins 1–16, 17–32, 33–64, 65–128.
Empty bins have zero count and null accuracy. Preserve every position's argmax
and the checkpoint hash. Report mean/min/max query accuracy across seeds, exact
accuracy mean, individual paired differences against MLP, and counts/accuracy by
distance pooled across seeds. Three seeds are descriptive, not enough for broad
significance claims. Uniform guessing over valid values has 6.25% expected query
accuracy; guessing over all logits has 1/41. These analytical baselines do not
account for learned value frequencies or context information.

After **all training** completes, load final checkpoints and measure inference
in inference_mode on the first 32 test sequences at each length and seed. Warm
up every method five times. Use Python Random(20260922) to permute four methods
per seed/length, then cyclically rotate that order for 32 rounds. Each method
occupies each timing position eight times. Save each order, duration and host
environment. Report mean of seed-wise median milliseconds and mean paired median
ratios to MLP. Timing excludes loading and preprocessing. No other experiment is
launched concurrently by this project; other host tenants remain uncontrolled.
Timing measures actual expanded operators, not best possible fused kernels.

## Verification and replay

Retain all original forward/gradient reference tests. Add an independent GELU
expression and per-projection loop for nonlinear forward/gradient checks,
finite-difference gradients, variable-length batch trimming, batched versus solo
outputs, causal/future perturbations, masked-key perturbations, pad gradients,
parameter counts, shared initialization, dictionary oracle and split disjointness.
Replay checks complete coverage (12 checkpoints, 36 test evaluations), every
manifest hash, regenerated datasets and order, oracle labels, initial shared
hashes, predictions exactly, losses within 1e-6, finite loss curves, final
validation metrics and balanced timing order. Verify from a wheel installed in
an isolated environment outside the source tree with no network during replay.
Different numerical libraries/CPUs can change float results; dependency versions
and hardware are preserved. Run without Python `-O` (assertions enforce checks).

## Related work and deviations

[Zoology (Arora et al.)](https://arxiv.org/abs/2312.04927) motivates multi-query
associative recall as a diagnostic. Its
[official generator](https://github.com/HazyResearch/zoology/blob/main/zoology/data/multiquery_ar.py)
was inspected on 2026-09-21: it uses a key/value prefix, query positions drawn from
a power-law distribution, and optional random non-query tokens. Here positions
are uniform, query keys are a subset of the definitions, distractors are a
separate alphabet, values may repeat, vocabulary is tiny, and training lengths
are mixed. This is **not a reproduction of published MQAR benchmark numbers**.
No source code or data are copied from that repository.

The [Transformer](https://arxiv.org/abs/1706.03762) supplies the common causal
sequence model and sinusoidal positions. The prior experiment's related-work
section discusses [Transformer in Transformer](https://arxiv.org/abs/2103.00112):
our inner vectors are learned projections of one token, not image subpatches,
and tied-QKV mean pooling differs from its inner transformer. We claim neither a
new operator nor an exhaustive literature survey.

Conclusions are limited to this generator, tiny models, three paired seeds,
this update budget and CPU implementation. Poor length transfer can reflect
position extrapolation and distractor shifts, not just recall capacity. Equal
update budgets favor neither equal wall time nor equal optimization difficulty.
A speed disadvantage, no gain over nonlinear pooling, saturation or failure to
learn all remain valid outcomes. No language-model or accelerator claim follows.
