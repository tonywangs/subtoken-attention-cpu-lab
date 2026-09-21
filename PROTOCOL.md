# Experimental protocol

## Precisely tested module

For normalized token vector x in R^d, independently learned affine maps produce
z_i = W_i x + b_i in R^r, i = 1,...,K. With Z stacking the z_i as rows:

```
A = softmax_rows(Z Z^T / sqrt(r))
u = (1/K) sum_i (A Z)_i
f(x) = W_out u + b_out
```

Here d=32, r=16, K=4. Queries, keys and values are all Z: there are **no additional
Q/K/V projections**. Each W_i is independent; the implementation packs them into
one linear layer, then reshapes to [batch,time,K,r]. Attention never spans batch
or sequence axes. K=1 reduces to a composition of two affine maps. No activation,
inner residual or inner normalization is added. The row softmax is the module's
nonlinearity.

Each of two transformer blocks computes:

```
h = x + causal_MHA(LayerNorm(x))
x_next = h + f(LayerNorm(h))
```

MHA uses four sequence heads, width 32, learned Q/K/V/output projections and
strict causal masking including the current position. Learned token and absolute
position embeddings precede the blocks; final LayerNorm and an untied 19-class
linear readout follow them. All linear maps have biases. LayerNorm uses PyTorch's
default epsilon 1e-5. No dropout. Maximum embedding length 16.

The MLP is Linear(32,41), exact GELU, Linear(41,32). The pooling ablation uses the
same projections as attention but u = mean_i(z_i). Pooling has exactly the same
parameter count as attention, but collapses algebraically to a single affine
transformation of rank at most 16. Thus this ablation removes both attention and
its nonlinearity; it does not isolate attention from generic nonlinear capacity.
No unused parameters are inserted to match counts.

Module counts: attention/pool K*r*(d+1)+d*(r+1)=2,656; MLP
41*(32+1)+32*(41+1)=2,697. Full-model counts are saved with results; matching is
asserted within 2%. Projection multiply-adds per token are K*d*r+r*d=2,560 for
pool, plus 2*K*K*r=512 for attention (excluding softmax), versus 2*d*41=2,624
for MLP (excluding GELU). Equal parameter counts do not imply equal computation.
Attention stores K^2 scores per token; ordinary O(T^2*d) sequence attention is
shared by all methods. Pooling is intentionally executed as expanded projections,
not algebraically fused, to preserve the ablation implementation.

## Data and tasks

Vocabulary: keys 0..7, values 8..15, BOS=16, separator=17; token 18 is unused but
included in the shared 19-class output vocabulary. Independent NumPy PCG64 RNGs
use dataset seeds 10011, 10022, 10033. Each task/seed has 4,096 training, 512
validation and 512 test examples, drawn in that order with rejection of duplicate
prompts across all three splits. Methods share exact arrays. Seeds define
independent replicates; overlap across different seeds is allowed. Each shard is
saved as compressed NPZ, and a SHA-256 over concatenated little-endian int64 X
then Y bytes identifies each split. Training sampling uses a separate Torch
Generator with seed 20000+model_seed, shared across methods.

* Recall: choose four ordered distinct keys uniformly without replacement from
  eight, four independent uniform values, and a uniform query index among the
  four pairs. Input is BOS,k1,v1,...,k4,v4,SEP,query (11 tokens). Supervise only
  the final position with its associated value. Other targets are -100 (ignored).
  Query is always present, values may repeat. This is fixed-length bounded recall,
  not long-context generalization. Identical pair lists with different queries
  are distinct prompts and may occur across splits.
* Copy: draw six independent uniform values. Input is
  BOS,v1,...,v6,SEP,v1,...,v5 (13 tokens); positions 7..12 predict v1..v6.
  Uniqueness is enforced on BOS,values,SEP, before the teacher-forced suffix.
  All other targets are -100. Evaluate both teacher-forced next-token predictions
  and autoregressive six-token generation from the eight-token prompt. No EOS or
  variable-length behavior is tested.

Uniform guesses over the eight valid values have expected 12.5% target-token
accuracy; a uniform 19-class readout has 1/19 accuracy. Report cross-entropy over
19 classes, target-token accuracy, and per-example exact match. Copy exact match
requires all six targets. The model has no access to test inputs during training.

## Budget and evaluation

The separate pilot uses seed 99001, 128/16/16 examples, batch 32, five warmup and
25 measured optimizer steps on each task/method. It selects a common integer
step budget min(1000, floor(1800/(18*slowest median step seconds))), at least one.
This caps predicted training at 30 minutes and leaves substantial time for checks.
Pilot losses are not used for selection; no test metrics are evaluated there.
Settings and raw times are preserved in configs/pilot. frozen.json is written
before final training and copied into final artifacts.

Training: paired model seeds 11,22,33, AdamW (lr .003, weight decay .01, default
betas .9/.999 and epsilon 1e-8), batch 32 sampled with replacement, global gradient
norm clipping at 1, float32, deterministic PyTorch algorithms, CPU only, one
intra-op and one inter-op thread. Same seed does not imply identical shared-layer
initial weights after differently shaped modules consume random draws. There is
no scheduling, early stopping, tuning or checkpoint selection: evaluate the final
step. Validation is diagnostic every 100 steps, at zero, and at the final step.
Each sampled-batch loss and pre-clipping gradient norm is saved. Training wall
time includes diagnostic validation and excludes checkpoint saving/test inference.

Test is evaluated once per final checkpoint in the experiment; later verification
replays the same evaluation without selecting settings. Inference timing uses
batch 32 and the task's teacher-forced length, five warmups and twenty timed calls
under inference_mode, without data loading. Raw samples and training wall time
are preserved. Fixed sequential execution order is MLP,pool,attention, so timing
is descriptive and can reflect host noise. Report medians and paired ratios;
three seeds are too few for broad statistical claims.

## Correctness and artifact checks

Unittests compare double-precision vectorized outputs and every input/parameter
gradient to a separately looped implementation for K=1 and K=4. Also included:
finite-difference input gradcheck, K=1 affine identity, batch/token perturbation
and gradient isolation, full-model causal/batch isolation for every architecture,
finite model gradients, parameter counts, dataset semantics and disjointness, and
the pooling affine identity. Saved artifact verification checks complete run
coverage, hashes, regenerated inputs, final checkpoints, all held-out predictions
and metrics, training curve length and timing sample count.

## Related work and scope

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduces the
Transformer's sequence attention and positionwise feed-forward structure. We
retain sequence attention and substitute only its subsequent per-token sublayer.
[Transformer in Transformer](https://arxiv.org/abs/2103.00112) attends among visual
words inside image patches and also models patch relationships. That hierarchy
is relevant prior art, but our inner objects are affine projections of the same
vector, not spatial subpatches. [FNet](https://arxiv.org/abs/2105.03824) replaces
sequence attention with Fourier mixing; we do not test that substitution.
The [PyTorch MultiheadAttention implementation](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/activation.py)
provides the standard sequence operator used here; the custom inner attention is
implemented directly with matrix multiplication and row softmax. These sources
motivate the comparison, not a claim of novelty or exhaustive prior-art coverage.

Conclusions apply only to this tied-QKV, mean-pooled within-token interpretation,
tiny two-layer models, two synthetic tasks, three seeds and this CPU environment.
They do not establish results for language, alternative inner attention designs,
large models, accelerators, length extrapolation, or equal wall-clock budgets.

The inspected [official TNT implementation](https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/tnt_pytorch/tnt.py)
uses learned Q/K and V maps, inner attention plus an inner MLP with residuals,
and a normalized flattened projection into outer patch tokens (Block.forward).
Our module omits those inner maps, residuals and MLP and mean-pools instead.
No TNT implementation code or weights are vendored.

Package preparation ran concurrently with some early training runs on the shared
host. Original per-run wall times therefore include possible resource contention;
they are not controlled performance benchmarks. A separate post-training timing
check, when present, is reported separately rather than replacing these samples.

The separate timing pass (`scripts/benchmark.py`) loads the same 18 checkpoints,
warms each up five times, then measures 30 rounds in independently shuffled order
using Python Random seed 20260921. Batch size remains 32. It runs after training
and package preparation, with no other compute job launched by this task. Order,
raw seconds, environment, and a digest of the source summary are saved in
`results/timing.json`. This is a measurement-quality check, not a budget change or
a model-selection experiment. It still cannot control other tenants on the host.
