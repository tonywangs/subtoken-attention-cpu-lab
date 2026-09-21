"""Verify frozen configuration linkage and the preserved experiment source hashes."""
import hashlib
import json
from pathlib import Path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

for path, expected in json.loads(Path('results/source-hashes.json').read_text()).items():
    assert sha(path) == expected, path
frozen = json.loads(Path('configs/pilot/frozen.json').read_text())
assert frozen == json.loads(Path('results/final/config.json').read_text())
assert frozen['pilot_sha256'] == sha('configs/pilot/pilot.json')
pilot = json.loads(Path('configs/pilot/pilot.json').read_text())
assert frozen['steps'] == pilot['selected_steps']
print('Frozen config, pilot linkage, and experiment source hashes verified.')
