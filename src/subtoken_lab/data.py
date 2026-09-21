import hashlib
import numpy as np
import torch


def arrays(task, seed, sizes=(4096, 512, 512)):
    rng = np.random.default_rng(seed)
    seen, rows, targets = set(), [], []
    while len(rows) < sum(sizes):
        if task == 'copy':
            values = rng.integers(8, 16, size=6).tolist()
            x = [16] + values + [17] + values[:-1]
            y = [-100] * 7 + values
        elif task == 'recall':
            keys = rng.choice(8, size=4, replace=False).tolist()
            values = rng.integers(8, 16, size=4).tolist()
            q = int(rng.integers(4))
            x = [16] + [v for pair in zip(keys, values) for v in pair] + [17, keys[q]]
            y = [-100] * 10 + [values[q]]
        else:
            raise ValueError(task)
        # Disjoint underlying prompts, not merely disjoint teacher-forced rows.
        key = tuple(x[:8] if task == 'copy' else x)
        if key not in seen:
            seen.add(key)
            rows.append(x)
            targets.append(y)
    x, y = np.array(rows, dtype='<i8'), np.array(targets, dtype='<i8')
    cuts = np.cumsum((0,) + tuple(sizes))
    return {name: (torch.from_numpy(x[a:b].copy()), torch.from_numpy(y[a:b].copy()))
            for name, a, b in zip(('train', 'validation', 'test'), cuts[:-1], cuts[1:])}


def digest(pair):
    return hashlib.sha256(b''.join(t.numpy().astype('<i8').tobytes() for t in pair)).hexdigest()
