"""Post-training, randomized-order CPU inference timings; no training or tuning."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path
import torch
from subtoken_lab.cli import setup, environment, sha, save
from subtoken_lab.data import arrays
from subtoken_lab.model import Model


def check(path, root):
    obj = json.loads(path.read_text())
    assert obj['summary_sha256'] == sha(root / 'summary.json')
    summary = json.loads((root / 'summary.json').read_text())
    expected = {r['name'] for r in summary}
    assert set(obj['seconds']) == expected
    assert all(len(v) == 30 and min(v) > 0 for v in obj['seconds'].values())
    assert len(obj['order']) == 30 * len(expected)
    for name in expected:
        assert obj['order'].count(name) == 30
    print('Post-training timing coverage and source digest verified.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/final'))
    p.add_argument('--output', type=Path, default=Path('results/timing.json'))
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if a.check:
        check(a.output, a.root)
        return
    if a.output.exists():
        raise FileExistsError(a.output)
    setup()
    config = json.loads((a.root / 'config.json').read_text())
    summary = json.loads((a.root / 'summary.json').read_text())
    models, batches = {}, {}
    for r in summary:
        name = r['name']
        model = Model(r['method']).eval()
        assert sha(a.root / f'{name}.pt') == r['checkpoint_sha256']
        model.load_state_dict(torch.load(a.root / f'{name}.pt', weights_only=True))
        models[name] = model
        batches[name] = arrays(r['task'], r['seed'] + config['dataset_seed_offset'], tuple(config['sizes']))['test'][0][:32]
    rng = random.Random(20260921)
    seconds = {name: [] for name in models}
    order = []
    with torch.inference_mode():
        for name, model in models.items():
            for _ in range(5):
                model(batches[name])
        for _ in range(30):
            names = list(models)
            rng.shuffle(names)
            for name in names:
                start = time.perf_counter()
                models[name](batches[name])
                seconds[name].append(time.perf_counter() - start)
                order.append(name)
    save(a.output, dict(environment=environment(), seed=20260921, warmup=5, batch=32,
                        rounds=30, order=order, seconds=seconds, summary_sha256=sha(a.root / 'summary.json')))
    check(a.output, a.root)
    print({name: round(statistics.median(v) * 1000, 3) for name, v in seconds.items()})


if __name__ == '__main__':
    main()
