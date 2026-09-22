"""Offline CLI: python -m subtoken_lab.mqar {pilot,freeze,run,verify,report}."""
import argparse
import datetime
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from .cli import save, sha, setup, environment, compare_metrics
from .mqar_data import datasets, generate, validate, digest, batch, FIELDS
from .mqar_model import RecallModel, METHODS

SOURCE_NAMES = ('mqar.py', 'mqar_data.py', 'mqar_model.py', 'model.py', 'cli.py')
DISTANCES = ((1, 16), (17, 32), (33, 64), (65, 128))


def sources():
    return {name: sha(Path(__file__).with_name(name)) for name in SOURCE_NAMES}


def base_config():
    return dict(schema=2, methods=list(METHODS), seeds=[111, 222, 333],
                train_size=8192, validation_size=256, test_size=512,
                train_lengths=[32, 40, 48], test_lengths=[48, 72, 96],
                batch=32, lr=0.003, weight_decay=0.01, clip=1., eval_interval=200,
                checkpoint_selection='final step only; validation diagnostic only',
                distance_bins=[list(b) for b in DISTANCES], timing_rounds=32, timing_warmup=5,
                timing_seed=20260922, pilot_seeds=[99001, 99002])


def check_config(c):
    for k, v in base_config().items():
        assert c[k] == v, k
    assert c['d'] in (32, 48) and 100 <= c['steps'] <= 2000
    assert c['source_sha256'] == sources(), 'Frozen experiment sources changed'


def train_step(model, opt, x, y, clip=1.):
    model.train()
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    opt.step()
    return loss.item(), norm.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    pred = np.full_like(data['y'], -100)
    loss_sum = 0.
    for start in range(0, len(pred), 64):
        stop = min(start + 64, len(pred))
        x, y = batch(data, slice(start, stop))
        logits = model(x)
        loss_sum += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum').item()
        pred[start:stop, :x.shape[1]] = logits.argmax(-1).numpy()
    mask = data['y'] != -100
    correct = pred == data['y']
    by_distance = {}
    for lo, hi in DISTANCES:
        selected = mask & (data['distance'] >= lo) & (data['distance'] <= hi)
        n = int(selected.sum())
        hits = int(correct[selected].sum())
        by_distance[f'{lo}-{hi}'] = dict(count=n, correct=hits, accuracy=hits / n if n else None)
    return dict(loss=loss_sum / int(mask.sum()), accuracy=float(correct[mask].mean()),
                exact_match=float((correct | ~mask).all(-1).mean()),
                queries=int(mask.sum()), sequences=len(pred), by_distance=by_distance), pred


def equal_metrics(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_metrics(a[k], b[k])
    elif a is None:
        assert b is None
    else:
        assert abs(a - b) <= 1e-6, (a, b)


def pilot(out):
    out.mkdir(parents=True, exist_ok=False)
    records = []
    c = base_config()
    # Timing-driven selection only; development accuracy is preserved, never optimized.
    for d in (32, 48):
        for seed in c['pilot_seeds']:
            data = datasets(c, seed, include_test=False)
            for method in METHODS:
                torch.manual_seed(seed)
                model = RecallModel(method, d)
                opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
                rng = np.random.default_rng(seed + 20000)
                samples, losses = [], []
                for step in range(100):
                    idx = rng.integers(0, c['train_size'], c['batch'])
                    x, y = batch(data['train'], idx)
                    t = time.perf_counter()
                    loss, _ = train_step(model, opt, x, y)
                    elapsed = time.perf_counter() - t
                    losses.append(loss)
                    if step >= 10:
                        samples.append(elapsed)
                metrics, _ = evaluate(model, data['validation'])
                records.append(dict(d=d, seed=seed, method=method, step_seconds=samples,
                                    train_losses=losses, validation=metrics,
                                    data_hashes={k: digest(v) for k, v in data.items()}))
                print('pilot', d, seed, method, statistics.median(samples), metrics['accuracy'], flush=True)
    costs = {d: max(statistics.median(r['step_seconds']) for r in records if r['d'] == d)
             for d in (32, 48)}
    # 30-minute estimated training budget, 25% headroom for diagnostics and noise.
    selected_d = 48 if 12 * 2000 * costs[48] * 1.25 <= 1800 else 32
    steps = max(100, min(2000, int(1800 / (12 * costs[selected_d] * 1.25))))
    save(out / 'pilot.json', dict(environment=environment(), records=records,
                                 selected_d=selected_d, selected_steps=steps,
                                 rule='d=48 if 12*2000*max_median_step(48)*1.25<=1800 else 32; steps=max(100,min(2000,floor(1800/(12*max_median_step(d)*1.25))))',
                                 source_sha256=sources()))


def freeze(pilot_path, protocol, out):
    if out.exists():
        raise FileExistsError(out)
    p = json.loads(pilot_path.read_text())
    assert p['source_sha256'] == sources()
    c = base_config()
    c.update(d=p['selected_d'], steps=p['selected_steps'], pilot_sha256=sha(pilot_path),
             protocol_sha256=sha(protocol), source_sha256=sources(),
             frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    check_config(c)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, c)
    print('Frozen', out, sha(out), flush=True)


def state_hash(model, shared=False):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        if shared and key.startswith('mix.'):
            continue
        h.update(key.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_model(c, out, seed, method):
    model = RecallModel(method, c['d'])
    model.load_state_dict(torch.load(out / f'{seed}-{method}.pt', map_location='cpu', weights_only=True))
    return model.eval()


def timing(c, out, all_data):
    records = []
    # Rotate a seeded permutation: every method occupies each order position equally often.
    rng = random.Random(c['timing_seed'])
    for seed in c['seeds']:
        models = {m: load_model(c, out, seed, m) for m in METHODS}
        for length in c['test_lengths']:
            x, _ = batch(all_data[seed][f'test{length}'], slice(0, c['batch']))
            order = list(METHODS)
            rng.shuffle(order)
            with torch.inference_mode():
                for m in order:
                    for _ in range(c['timing_warmup']):
                        models[m](x)
                for round_idx in range(c['timing_rounds']):
                    rotated = order[round_idx % 4:] + order[:round_idx % 4]
                    for position, method in enumerate(rotated):
                        start = time.perf_counter()
                        models[method](x)
                        seconds = time.perf_counter() - start
                        records.append(dict(seed=seed, length=length, round=round_idx,
                                            position=position, method=method, seconds=seconds))
    return dict(environment=environment(), batch=c['batch'], rounds=c['timing_rounds'],
                warmup=c['timing_warmup'], records=records)


def run(config, protocol, pilot_path, out):
    c = json.loads(config.read_text())
    check_config(c)
    assert sha(protocol) == c['protocol_sha256']
    assert sha(pilot_path) == c['pilot_sha256']
    out.mkdir(parents=True, exist_ok=False)
    # Preserve exact pre-test freeze and its inputs inside the artifact directory.
    (out / 'config.json').write_bytes(config.read_bytes())
    (out / 'protocol.md').write_bytes(protocol.read_bytes())
    (out / 'pilot.json').write_bytes(pilot_path.read_bytes())
    save(out / 'environment.json', environment())
    save(out / 'started.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   config_sha256=sha(config), source_sha256=sources()))
    counts = {m: sum(p.numel() for p in RecallModel(m, c['d']).parameters()) for m in METHODS}
    assert max(counts.values()) / min(counts.values()) < 1.02
    save(out / 'parameters.json', counts)
    summary, all_data = [], {}
    for seed in c['seeds']:
        data = datasets(c, seed)
        all_data[seed] = data
        seen = set()
        for split, shard in data.items():
            validate(shard, seen)
            np.savez_compressed(out / f'{seed}-{split}.npz', **shard)
        save(out / f'{seed}-data.json', {k: digest(v) for k, v in data.items()})
        shared_hash = None
        # Rotate training order across seeds; inference timing is balanced separately.
        offset = c['seeds'].index(seed)
        method_order = list(METHODS[offset:] + METHODS[:offset])
        for method in method_order:
            name = f'{seed}-{method}'
            torch.manual_seed(seed)
            model = RecallModel(method, c['d'])
            initial_hash = state_hash(model, shared=True)
            if shared_hash is None:
                shared_hash = initial_hash
            assert shared_hash == initial_hash
            opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
            rng = np.random.default_rng(seed + 20000)
            order_hash = hashlib.sha256()
            curves = []
            start = time.perf_counter()
            for step in range(c['steps'] + 1):
                if step % c['eval_interval'] == 0 or step == c['steps']:
                    metrics, _ = evaluate(model, data['validation'])
                    curves.append(dict(step=step, validation=metrics))
                    print(name, 'step', step, 'validation', round(metrics['accuracy'], 4), flush=True)
                if step == c['steps']:
                    break
                indices = rng.integers(0, c['train_size'], c['batch'])
                order_hash.update(indices.astype('<i8').tobytes())
                x, y = batch(data['train'], indices)
                loss, norm = train_step(model, opt, x, y, c['clip'])
                curves.append(dict(step=step + 1, train_loss=loss, gradient_norm=norm))
            train_seconds = time.perf_counter() - start
            torch.save(model.state_dict(), out / f'{name}.pt')
            save(out / f'{name}-curves.json', curves)
            tests, predictions = {}, {}
            for length in c['test_lengths']:
                split = f'test{length}'
                tests[split], predictions[split] = evaluate(model, data[split])
            np.savez_compressed(out / f'{name}-predictions.npz', **predictions)
            summary.append(dict(seed=seed, method=method, parameters=counts[method], test=tests,
                                training_seconds=train_seconds, initial_shared_sha256=initial_hash,
                                training_order_sha256=order_hash.hexdigest(), checkpoint_sha256=sha(out / f'{name}.pt')))
            save(out / 'summary.json', summary)
            print(name, 'finished', {k: v['accuracy'] for k, v in tests.items()}, flush=True)
    save(out / 'timing.json', timing(c, out, all_data))
    save(out / 'completed.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    save(out / 'manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})


def verify(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    assert set(manifest) == {p.name for p in out.iterdir() if p.is_file()} - {'manifest.json'}
    for name, expected in manifest.items():
        assert Path(name).name == name and not (out / name).is_symlink()
        assert sha(out / name) == expected, name
    c = json.loads((out / 'config.json').read_text())
    check_config(c)
    assert sha(out / 'protocol.md') == c['protocol_sha256']
    assert sha(out / 'pilot.json') == c['pilot_sha256']
    p = json.loads((out / 'pilot.json').read_text())
    assert (p['selected_d'], p['selected_steps']) == (c['d'], c['steps'])
    started = json.loads((out / 'started.json').read_text())
    assert started['source_sha256'] == sources()
    assert started['config_sha256'] == sha(out / 'config.json')
    assert c['frozen_utc'] < started['utc'] < json.loads((out / 'completed.json').read_text())['utc']
    summary = json.loads((out / 'summary.json').read_text())
    assert len(summary) == 12
    assert {(r['seed'], r['method']) for r in summary} == {(s, m) for s in c['seeds'] for m in METHODS}
    counts = json.loads((out / 'parameters.json').read_text())
    assert max(counts.values()) / min(counts.values()) < 1.02
    for seed in c['seeds']:
        data, seen = datasets(c, seed), set()
        assert json.loads((out / f'{seed}-data.json').read_text()) == {k: digest(v) for k, v in data.items()}
        for split, shard in data.items():
            validate(shard, seen)
            with np.load(out / f'{seed}-{split}.npz', allow_pickle=False) as saved:
                assert set(saved.files) == set(FIELDS)
                for key in FIELDS:
                    np.testing.assert_array_equal(shard[key], saved[key])
        rows = [r for r in summary if r['seed'] == seed]
        assert len({r['initial_shared_sha256'] for r in rows}) == 1
        assert len({r['training_order_sha256'] for r in rows}) == 1
        order_hash, rng = hashlib.sha256(), np.random.default_rng(seed + 20000)
        for _ in range(c['steps']):
            order_hash.update(rng.integers(0, c['train_size'], c['batch']).astype('<i8').tobytes())
        for row in rows:
            name, method = f"{seed}-{row['method']}", row['method']
            assert order_hash.hexdigest() == row['training_order_sha256']
            torch.manual_seed(seed)
            assert state_hash(RecallModel(method, c['d']), shared=True) == row['initial_shared_sha256']
            model = load_model(c, out, seed, method)
            assert sum(p.numel() for p in model.parameters()) == row['parameters'] == counts[method]
            assert sha(out / f'{name}.pt') == row['checkpoint_sha256']
            with np.load(out / f'{name}-predictions.npz', allow_pickle=False) as predictions:
                assert set(predictions.files) == {f'test{n}' for n in c['test_lengths']}
                for length in c['test_lengths']:
                    split = f'test{length}'
                    metrics, pred = evaluate(model, data[split])
                    np.testing.assert_array_equal(pred, predictions[split])
                    equal_metrics(metrics, row['test'][split])
            curves = json.loads((out / f'{name}-curves.json').read_text())
            train = [r for r in curves if 'train_loss' in r]
            assert [r['step'] for r in train] == list(range(1, c['steps'] + 1))
            assert all(np.isfinite(r['train_loss']) and np.isfinite(r['gradient_norm']) for r in train)
            assert curves[-1]['step'] == c['steps'] and 'validation' in curves[-1]
            metrics, _ = evaluate(model, data['validation'])
            equal_metrics(metrics, curves[-1]['validation'])
            assert row['training_seconds'] > 0
            print('verified', name, flush=True)
    times = json.loads((out / 'timing.json').read_text())
    assert times['batch'] == c['batch'] and times['rounds'] == c['timing_rounds']
    assert times['warmup'] == c['timing_warmup']
    assert len(times['records']) == 3 * 3 * 4 * c['timing_rounds']
    rng = random.Random(c['timing_seed'])
    expected = []
    for seed in c['seeds']:
        for length in c['test_lengths']:
            order = list(METHODS)
            rng.shuffle(order)
            for rnd in range(c['timing_rounds']):
                rotated = order[rnd % 4:] + order[:rnd % 4]
                expected.extend((seed, length, rnd, pos, method) for pos, method in enumerate(rotated))
    actual = [(r['seed'], r['length'], r['round'], r['position'], r['method']) for r in times['records']]
    assert actual == expected
    assert all(np.isfinite(r['seconds']) and r['seconds'] > 0 for r in times['records'])
    print('Verified 12 checkpoints, 36 evaluations, dataset oracles, pairing, freeze, hashes and balanced timing.')


def report(out):
    c = json.loads((out / 'config.json').read_text())
    rows = json.loads((out / 'summary.json').read_text())
    times = json.loads((out / 'timing.json').read_text())['records']
    lines = ['# Variable-length recall results', '',
             'Generated from preserved artifacts; accuracy is the unweighted mean across three seeds.', '',
             f"Frozen configuration SHA-256: `{sha(out / 'config.json')}`.",
             f"Width {c['d']}; {c['steps']} updates per run; 512 sequences / 2,048 queries per seed and length.", '',
             '| Length | Method | Query accuracy % (seed min–max) | Exact sequence % | Inference ms / batch 32 | Ratio to MLP |',
             '|---:|---|---:|---:|---:|---:|']
    for length in c['test_lengths']:
        for method in METHODS:
            selected = [r['test'][f'test{length}'] for r in rows if r['method'] == method]
            accuracy = [r['accuracy'] * 100 for r in selected]
            latencies, ratios = [], []
            for seed in c['seeds']:
                med = lambda m: statistics.median(r['seconds'] for r in times if r['seed'] == seed and r['length'] == length and r['method'] == m)
                latencies.append(med(method) * 1000)
                ratios.append(med(method) / med('mlp'))
            lines.append(f"| {length} | {method} | {statistics.mean(accuracy):.2f} ({min(accuracy):.2f}–{max(accuracy):.2f}) | {statistics.mean(r['exact_match'] for r in selected)*100:.2f} | {statistics.mean(latencies):.3f} | {statistics.mean(ratios):.3f} |")
    lines += ['', '## Paired query accuracy differences', '', 'Percentage points relative to MLP; seed order 111, 222, 333. No significance claim.', '',
              '| Length | Method | Seed differences (pp) | Mean (pp) |', '|---:|---|---|---:|']
    for length in c['test_lengths']:
        for method in METHODS[1:]:
            values = []
            for seed in c['seeds']:
                acc = lambda m: next(r['test'][f'test{length}']['accuracy'] for r in rows if r['method'] == m and r['seed'] == seed)
                values.append(100 * (acc(method) - acc('mlp')))
            lines.append(f"| {length} | {method} | {', '.join(f'{v:+.2f}' for v in values)} | {statistics.mean(values):+.2f} |")
    lines += ['', '## Retrieval distance', '', 'Query position minus the corresponding definition value position. Counts pool three seeds.', '',
              '| Length | Method | Distance | Queries | Accuracy % |', '|---:|---|---|---:|---:|']
    for length in c['test_lengths']:
        for method in METHODS:
            selected = [r['test'][f'test{length}']['by_distance'] for r in rows if r['method'] == method]
            for lo, hi in DISTANCES:
                key = f'{lo}-{hi}'
                n = sum(r[key]['count'] for r in selected)
                hits = sum(r[key]['correct'] for r in selected)
                if n:
                    lines.append(f'| {length} | {method} | {key} | {n} | {100*hits/n:.2f} |')
    lines += ['', '## Training cost', '', 'Wall time includes validation; sequential runs on a shared host are not controlled timing comparisons.', '',
              '| Method | Parameters | Total training seconds (3 seeds) |', '|---|---:|---:|']
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        lines.append(f"| {method} | {selected[0]['parameters']} | {sum(r['training_seconds'] for r in selected):.2f} |")
    lines += ['', 'Uniform valid-value guessing has expected query accuracy 6.25%; uniform full-vocabulary guessing 2.44%.',
              'These are analytical baselines, not measured trained models. Four independent uniform value guesses give 0.001526% expected exact accuracy.', '',
              'See [protocol and limitations](MQAR_PROTOCOL.md) and [interpretation](MQAR_INTERPRETATION.md).', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pilot')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('verify')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('report')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--check', type=Path)
    args = parser.parse_args()
    setup()
    if args.command == 'pilot':
        pilot(args.out)
    elif args.command == 'freeze':
        freeze(args.pilot, args.protocol, args.out)
    elif args.command == 'run':
        run(args.config, args.protocol, args.pilot, args.out)
    elif args.command == 'verify':
        verify(args.out)
    else:
        rendered = report(args.out)
        if args.check:
            assert args.check.read_text() == rendered
            print('Report matches preserved artifacts.')
        else:
            print(rendered, end='')


if __name__ == '__main__':
    main()
