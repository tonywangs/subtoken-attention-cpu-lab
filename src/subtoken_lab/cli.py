import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import arrays, digest
from .model import Model

METHODS = ('mlp', 'pool', 'attention')


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def environment():
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(), cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(),
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                torch_config=torch.__config__.show())


@torch.no_grad()
def evaluate(model, pair):
    model.eval()
    x, y = pair
    logits = torch.cat([model(batch) for batch in x.split(64)])
    pred = logits.argmax(-1)
    mask = y != -100
    accuracy = (pred[mask] == y[mask]).float().mean().item()
    exact = ((pred == y) | ~mask).all(-1).float().mean().item()
    return dict(loss=F.cross_entropy(logits.flatten(0, 1), y.flatten()).item(),
                accuracy=accuracy, exact_match=exact), pred


@torch.no_grad()
def generated_copy(model, pair):
    model.eval()
    x, y = pair
    predictions = []
    for batch in x.split(64):
        prompt = batch[:, :8]
        outputs = []
        for _ in range(6):
            token = model(prompt)[:, -1].argmax(-1)
            outputs.append(token)
            prompt = torch.cat((prompt, token[:, None]), 1)
        predictions.append(torch.stack(outputs, 1))
    pred = torch.cat(predictions)
    truth = y[:, 7:]
    return dict(accuracy=(pred == truth).float().mean().item(),
                exact_match=(pred == truth).all(-1).float().mean().item()), pred


def pilot(out):
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for task in ('recall', 'copy'):
        x, y = arrays(task, 99001, (128, 16, 16))['train']
        for method in METHODS:
            torch.manual_seed(99001)
            model = Model(method)
            opt = torch.optim.AdamW(model.parameters(), lr=0.003)
            samples = []
            for step in range(30):
                start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[:32]).flatten(0, 1), y[:32].flatten())
                loss.backward()
                opt.step()
                if step >= 5:
                    samples.append(time.perf_counter() - start)
            results.append(dict(task=task, method=method, seconds_per_step=samples))
    slowest = max(statistics.median(r['seconds_per_step']) for r in results)
    # Spend at most 30 estimated CPU minutes on 18 training runs, capped at 1000 steps.
    steps = max(1, min(1000, int(1800 / (18 * slowest))))
    save(out / 'pilot.json', dict(environment=environment(), seed=99001, batch=32,
                                 warmup=5, measured=25, results=results, selected_steps=steps,
                                 rule='min(1000, floor(1800 / (18 * slowest median step seconds))), at least 1'))
    config = dict(schema=1, methods=list(METHODS), tasks=['recall', 'copy'], seeds=[11, 22, 33],
                  dataset_seed_offset=10000, sizes=[4096, 512, 512], steps=steps, batch=32,
                  lr=0.003, weight_decay=0.01, clip=1.0, eval_interval=100,
                  checkpoint_selection='final step; validation diagnostic only',
                  model=dict(d=32, r=16, k=4, layers=2, heads=4, mlp_hidden=41),
                  pilot_sha256=sha(out / 'pilot.json'))
    save(out / 'frozen.json', config)
    print(json.dumps(config, indent=2))


def validate_config(c):
    assert c['schema'] == 1
    assert c['model'] == dict(d=32, r=16, k=4, layers=2, heads=4, mlp_hidden=41)
    assert c['methods'] == list(METHODS)
    assert c['tasks'] == ['recall', 'copy']
    assert c['seeds'] == [11, 22, 33]
    assert len(c['sizes']) == 3 and all(n > 0 for n in c['sizes'])
    assert c['steps'] > 0 and c['batch'] > 0


def run(config_path, out):
    c = json.loads(config_path.read_text())
    validate_config(c)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', c)
    save(out / 'environment.json', environment())
    counts = {m: sum(p.numel() for p in Model(m).parameters()) for m in METHODS}
    assert max(counts.values()) / min(counts.values()) < 1.02
    save(out / 'parameters.json', counts)
    all_results = []
    for task in c['tasks']:
        for seed in c['seeds']:
            data = arrays(task, seed + c['dataset_seed_offset'], tuple(c['sizes']))
            hashes = {split: digest(pair) for split, pair in data.items()}
            save(out / f'{task}-{seed}-data.json', hashes)
            # Exact inputs saved in compressed shards; generation is separately verified.
            np.savez_compressed(out / f'{task}-{seed}-data.npz',
                                **{f'{split}_{axis}': t.numpy() for split, pair in data.items()
                                   for axis, t in zip(('x', 'y'), pair)})
            for method in c['methods']:
                name = f'{task}-{seed}-{method}'
                torch.manual_seed(seed)
                model = Model(method)
                generator = torch.Generator().manual_seed(seed + 20000)
                opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
                curves = []
                start = time.perf_counter()
                for step in range(c['steps'] + 1):
                    if step % c['eval_interval'] == 0 or step == c['steps']:
                        metrics, _ = evaluate(model, data['validation'])
                        curves.append(dict(step=step, validation=metrics))
                    if step == c['steps']:
                        break
                    model.train()
                    idx = torch.randint(len(data['train'][0]), (c['batch'],), generator=generator)
                    x, y = (t[idx] for t in data['train'])
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss: {name}, {step}')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['clip'], error_if_nonfinite=True)
                    opt.step()
                    curves.append(dict(step=step + 1, train_loss=loss.item(), gradient_norm=norm.item()))
                train_seconds = time.perf_counter() - start
                torch.save(model.state_dict(), out / f'{name}.pt')
                save(out / f'{name}-curves.json', curves)
                test, pred = evaluate(model, data['test'])
                prediction_data = dict(teacher_forced=pred.numpy())
                if task == 'copy':
                    test['autoregressive'], generated = generated_copy(model, data['test'])
                    prediction_data['autoregressive'] = generated.numpy()
                np.savez_compressed(out / f'{name}-predictions.npz', **prediction_data)
                samples = []
                model.eval()
                with torch.inference_mode():
                    batch = data['test'][0][:32]
                    for i in range(25):
                        t = time.perf_counter()
                        model(batch)
                        if i >= 5:
                            samples.append(time.perf_counter() - t)
                result = dict(name=name, task=task, seed=seed, method=method, test=test,
                              train_seconds=train_seconds, inference_batch=32, inference_warmup=5,
                              inference_seconds=samples, parameters=counts[method],
                              checkpoint_sha256=sha(out / f'{name}.pt'))
                all_results.append(result)
                save(out / 'summary.json', all_results)
                print(name, test, flush=True)
    manifest = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    save(out / 'manifest.json', manifest)


def verify(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    assert set(manifest) == {p.name for p in out.iterdir() if p.is_file()} - {'manifest.json'}
    for name, expected in manifest.items():
        assert sha(out / name) == expected, name
    c = json.loads((out / 'config.json').read_text())
    validate_config(c)
    summary = json.loads((out / 'summary.json').read_text())
    expected_runs = {(t, s, m) for t in c['tasks'] for s in c['seeds'] for m in c['methods']}
    assert {(r['task'], r['seed'], r['method']) for r in summary} == expected_runs
    assert len(summary) == len(expected_runs)
    for task in c['tasks']:
        for seed in c['seeds']:
            data = arrays(task, seed + c['dataset_seed_offset'], tuple(c['sizes']))
            saved = np.load(out / f'{task}-{seed}-data.npz', allow_pickle=False)
            assert json.loads((out / f'{task}-{seed}-data.json').read_text()) == {k: digest(v) for k, v in data.items()}
            for split, pair in data.items():
                for axis, t in zip(('x', 'y'), pair):
                    np.testing.assert_array_equal(saved[f'{split}_{axis}'], t.numpy())
            for r in [r for r in summary if r['task'] == task and r['seed'] == seed]:
                name = r['name']
                assert sha(out / f'{name}.pt') == r['checkpoint_sha256']
                model = Model(r['method'])
                model.load_state_dict(torch.load(out / f'{name}.pt', weights_only=True, map_location='cpu'))
                assert sum(p.numel() for p in model.parameters()) == r['parameters']
                metrics, pred = evaluate(model, data['test'])
                saved_pred = np.load(out / f'{name}-predictions.npz', allow_pickle=False)
                np.testing.assert_array_equal(pred.numpy(), saved_pred['teacher_forced'])
                if task == 'copy':
                    metrics['autoregressive'], generated = generated_copy(model, data['test'])
                    np.testing.assert_array_equal(generated.numpy(), saved_pred['autoregressive'])
                compare_metrics(metrics, r['test'])
                assert len(r['inference_seconds']) == 20 and min(r['inference_seconds']) > 0
                curves = json.loads((out / f'{name}-curves.json').read_text())
                assert sum('train_loss' in row for row in curves) == c['steps']
                assert curves[-1]['step'] == c['steps'] and 'validation' in curves[-1]
    print(f'Verified hashes, regenerated datasets, and checkpoint evaluations for {len(summary)} runs.')


def compare_metrics(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], dict):
            compare_metrics(actual[key], expected[key])
        else:
            assert abs(actual[key] - expected[key]) <= 1e-6, (key, actual[key], expected[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pilot')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('verify')
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    setup()
    if args.command == 'pilot':
        pilot(args.out)
    elif args.command == 'run':
        run(args.config, args.out)
    else:
        verify(args.out)


if __name__ == '__main__':
    main()
