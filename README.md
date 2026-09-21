# Within-token attention CPU lab

A reproducible comparison of attention among learned projections of **one token**,
a parameter-matched MLP, and projection/mean pooling. Ordinary causal sequence
attention is retained. This is a small synthetic experiment, not a new language
model architecture claim. See [the protocol](PROTOCOL.md) and [results](RESULTS.md).

## Install and use

Python 3.10+ with a CPU PyTorch installation is required. The recorded experiment
uses Python 3.12, PyTorch 2.6.0+cpu and NumPy 2.2.6. On Debian, install the Python
venv package if `python3 -m venv` is unavailable.

```sh
python3 -m venv .venv
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy==2.2.6
.venv/bin/pip install .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/subtoken-lab verify --out results/final
```

`verify` is offline after installation: it checks every artifact hash, regenerates
all input shards, loads all 18 checkpoints, and recomputes test metrics and
predictions, including autoregressive copy. Floating point loss tolerance is
1e-6; predicted token IDs must match exactly. Different PyTorch/CPU versions can
change numerical behavior. Use the saved dependency versions for closest replay.

To reproduce training with the frozen settings (output must be a new directory):

```sh
.venv/bin/subtoken-lab run --config configs/pilot/frozen.json --out /tmp/subtoken-reproduction
.venv/bin/subtoken-lab verify --out /tmp/subtoken-reproduction
```

To repeat the budget-selection procedure separately:

```sh
.venv/bin/subtoken-lab pilot --out /tmp/subtoken-new-pilot
```

The pilot is a timing exercise, not a hyperparameter or test-accuracy search.
The saved final suite uses its already frozen configuration. No paid services,
private data, network access, or accelerators are needed to run the experiment.
Checkpoint files contain only model weights; there is no optimizer-resume command.

The full resolved dependency list is in `results/dependencies.txt`. For an offline
installation, first download those wheels and the project wheel on a connected
machine, then install with pip's `--no-index --find-links` options. Dependency
wheels are intentionally not stored in this repository.

`python3 scripts/report.py --check` verifies that the human-readable tables match
the raw results. `python3 scripts/check_tree.py` checks repository artifact size
and file-count limits. The manifest detects accidental changes; it is not a signed
attestation and must be obtained from a trusted checkout. Run verification without
Python's `-O` flag, because invariant checks use assertions.

For the separate timing experiment:

```sh
.venv/bin/python scripts/benchmark.py --output /tmp/subtoken-new-timing.json
.venv/bin/python scripts/benchmark.py --check
python3 scripts/check_protocol.py
```

The default timing output is immutable: use a new path to repeat measurements.
`check_protocol.py` verifies that the final configuration matches the pilot's
frozen settings and that the preserved experiment sources have not changed.
