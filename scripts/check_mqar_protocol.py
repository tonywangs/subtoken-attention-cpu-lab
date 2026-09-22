"""Audit pre-test freeze, pilot selection arithmetic, source and artifact linkage."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


c = read('configs/mqar-frozen.json')
p = read('configs/mqar-pilot/pilot.json')
assert c == read('results/mqar/config.json')
assert sha('configs/mqar-frozen.json') == sha('results/mqar/config.json')
assert sha('MQAR_PROTOCOL.md') == c['protocol_sha256'] == sha('results/mqar/protocol.md')
assert sha('configs/mqar-pilot/pilot.json') == c['pilot_sha256'] == sha('results/mqar/pilot.json')
assert p['source_sha256'] == c['source_sha256']
for name, expected in c['source_sha256'].items():
    assert sha('src/subtoken_lab/' + name) == expected, name
rows = p['records']
assert len(rows) == 16
assert {(r['d'], r['seed'], r['method']) for r in rows} == {
    (d, s, m) for d in (32, 48) for s in (99001, 99002)
    for m in ('mlp', 'pool', 'nonlinear', 'attention')}
assert all(len(r['step_seconds']) == 90 and min(r['step_seconds']) > 0 for r in rows)
assert all(len(r['train_losses']) == 100 for r in rows)
assert set(c['seeds']).isdisjoint(c['pilot_seeds'])
for seed in c['pilot_seeds']:
    hashes = [r['data_hashes'] for r in rows if r['seed'] == seed]
    assert all(h == hashes[0] for h in hashes)
    assert all(set(h) == {'train', 'validation'} for h in hashes)
cost = {d: max(statistics.median(r['step_seconds']) for r in rows if r['d'] == d) for d in (32, 48)}
width = 48 if 12 * 2000 * cost[48] * 1.25 <= 1800 else 32
steps = max(100, min(2000, int(1800 / (12 * cost[width] * 1.25))))
assert (width, steps) == (p['selected_d'], p['selected_steps']) == (c['d'], c['steps'])
assert c['test_size'] >= 512 and c['test_lengths'] == [48, 72, 96]
assert c['train_lengths'] == [32, 40, 48]
started = read('results/mqar/started.json')
assert started['config_sha256'] == sha('configs/mqar-frozen.json')
assert c['frozen_utc'] < started['utc'] < read('results/mqar/completed.json')['utc']
print(f'Frozen source/protocol/pilot linkage and timing-based selection verified: d={width}, steps={steps}.')
