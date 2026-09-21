"""Render factual tables from the preserved run summaries (standard library only)."""
import argparse
import json
import statistics as st
from pathlib import Path


def report(root):
    runs = json.loads((root / 'summary.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    counts = json.loads((root / 'parameters.json').read_text())
    env = json.loads((root / 'environment.json').read_text())
    lines = ['# Recorded results', '',
             f"All {len(runs)} runs used {config['steps']} optimizer steps, batch {config['batch']}, and the frozen configuration.",
             f"CPU: {env['cpu'].lstrip(': ')}; PyTorch {env['torch']}; {env['threads']} compute thread.",
             'No configuration changes or checkpoint selection used final test results.', '',
             f"Parameters: MLP {counts['mlp']:,}; pool and attention {counts['attention']:,}. "
             f"Maximum difference: {(max(counts.values())/min(counts.values())-1)*100:.3f}%.", '',
             '## Each seed', '',
             'Accuracy and exact match are percentages. TF = teacher forced; AR = autoregressive.',
             'Recall has one target, so token accuracy equals exact match. Timing is batch-32 median milliseconds.', '',
             '| Task | Seed | Method | Test CE | TF accuracy | TF exact | AR accuracy | AR exact | Inference ms | Train s |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        t = r['test']
        ar = t.get('autoregressive')
        ar_text = f"{100*ar['accuracy']:.2f} | {100*ar['exact_match']:.2f}" if ar else '— | —'
        lines.append(f"| {r['task']} | {r['seed']} | {r['method']} | {t['loss']:.5f} | "
                     f"{100*t['accuracy']:.2f} | {100*t['exact_match']:.2f} | {ar_text} | "
                     f"{1000*st.median(r['inference_seconds']):.3f} | {r['train_seconds']:.2f} |")
    if all(r['test']['accuracy'] == 1.0 and
           r['test'].get('autoregressive', {'exact_match': 1.0})['exact_match'] == 1.0 for r in runs):
        lines += ['', 'Every final run reached 100% held-out target accuracy and exact match, including',
                  'autoregressive copy. These tasks saturated: this suite finds no final-accuracy benefit',
                  'from within-token attention. Loss and timing still differ, but ceiling accuracy limits',
                  'the experiment’s ability to distinguish representation quality.']
    lines += ['', '## Across seeds', '', 'Mean ± sample standard deviation over three seeds; not confidence intervals.', '',
              '| Task | Method | Test CE | TF accuracy % | TF exact % | AR exact % | Inference ms |',
              '|---|---|---:|---:|---:|---:|---:|']
    def fmt(values, scale=100):
        return f'{scale*st.mean(values):.2f} ± {scale*st.stdev(values):.2f}'
    for task in config['tasks']:
        for method in config['methods']:
            subset = [r for r in runs if r['task'] == task and r['method'] == method]
            losses = [r['test']['loss'] for r in subset]
            lines.append(f"| {task} | {method} | {st.mean(losses):.5f} ± {st.stdev(losses):.5f} | "
                         f"{fmt([r['test']['accuracy'] for r in subset])} | "
                         f"{fmt([r['test']['exact_match'] for r in subset])} | "
                         + (fmt([r['test']['autoregressive']['exact_match'] for r in subset]) if task == 'copy' else '—')
                         + f" | {fmt([st.median(r['inference_seconds']) for r in subset], 1000)} |")
    lines += ['', '## Paired differences and costs', '',
              'Each ratio divides the method by the MLP at the same task/seed. Values are mean ± sample SD.',
              'Accuracy difference is in percentage points; ratios greater than one mean slower.', '',
              '| Task | Method | TF accuracy difference | Inference ratio | Training ratio |',
              '|---|---|---:|---:|---:|']
    for task in config['tasks']:
        for method in ('pool', 'attention'):
            diffs, inference, train = [], [], []
            for seed in config['seeds']:
                base = next(r for r in runs if (r['task'], r['seed'], r['method']) == (task, seed, 'mlp'))
                r = next(r for r in runs if (r['task'], r['seed'], r['method']) == (task, seed, method))
                diffs.append(r['test']['accuracy'] - base['test']['accuracy'])
                inference.append(st.median(r['inference_seconds']) / st.median(base['inference_seconds']))
                train.append(r['train_seconds'] / base['train_seconds'])
            lines.append(f'| {task} | {method} | {fmt(diffs)} | {fmt(inference, 1)} | {fmt(train, 1)} |')
    timing_path = root.parent / 'timing.json'
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())['seconds']
        lines += ['', '## Separate post-training timing pass', '',
                  'Thirty randomized rounds over all 18 checkpoints, after five warmups per checkpoint.',
                  'No package installation or training was running in this task during this pass.',
                  'Shared-host noise remains possible. Batch size is 32; values summarize three seed medians.', '',
                  '| Task | Method | Inference ms (mean ± SD) | Paired ratio to MLP (mean ± SD) |',
                  '|---|---|---:|---:|']
        for task in config['tasks']:
            for method in config['methods']:
                medians = [st.median(timing[f'{task}-{s}-{method}']) for s in config['seeds']]
                ratios = [value / st.median(timing[f'{task}-{s}-mlp'])
                          for s, value in zip(config['seeds'], medians)]
                lines.append(f'| {task} | {method} | {fmt(medians, 1000)} | {fmt(ratios, 1)} |')
    lines += ['', '## Reading these results', '',
              'The tables preserve every final run; no runs were discarded. High copy accuracy at this',
              'single fixed length does not demonstrate length generalization. Ceiling scores leave open',
              'how these methods behave on harder inputs. Three seeds support descriptive comparisons only.',
              'Pooling is an affine, low-rank ablation; its differences',
              'cannot be attributed solely to the absence of attention. No claim of general improvement is made.', '',
              'Raw losses, gradient norms, validation curves, held-out predictions, input shards, checkpoints,',
              'hashes, and twenty inference timing samples per run are in `results/final/`.',
              'See [PROTOCOL.md](PROTOCOL.md) for generation rules, architecture, prior work and limitations.',
              'This document is reproduced by `python3 scripts/report.py --check`.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/final'))
    p.add_argument('--output', type=Path, default=Path('RESULTS.md'))
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    text = report(args.root)
    if args.check:
        assert args.output.read_text() == text, 'Report differs from source artifacts'
        print('Report matches preserved results.')
    else:
        args.output.write_text(text)
