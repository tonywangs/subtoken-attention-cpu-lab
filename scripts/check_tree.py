"""Check the publishable file set and emit a content digest without staging files."""
import hashlib
import json
import subprocess
from pathlib import Path

raw = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
files = sorted(set(s.decode() for s in raw.split(b'\0') if s))
assert len(files) <= 1000, len(files)
records = []
for name in files:
    p = Path(name)
    assert p.is_file() and not p.is_symlink(), name
    assert p.stat().st_size <= 10 * 1024**2, name
    assert not name.endswith('.log') or name == 'results/tests.log', name
    assert not any(part in ('.agents', '.codex', '.venv') for part in p.parts), name
    records.append((name, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()))
size = sum(r[1] for r in records)
assert size <= 32 * 1024**2, size
print(json.dumps(dict(files=len(files), bytes=size,
                     content_sha256=hashlib.sha256(json.dumps(records).encode()).hexdigest()), indent=2))
