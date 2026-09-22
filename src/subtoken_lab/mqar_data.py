"""CPU-scale MQAR variant. No labels or previous answers occur in query suffixes."""
import hashlib
import numpy as np
import torch

PAD, VOCAB, PAIRS, QUERIES = 0, 41, 6, 4
FIELDS = ('x', 'y', 'distance', 'length')


def generate(seed, count, lengths, seen=None):
    if count < 1 or not lengths or min(lengths) < 2 * PAIRS + QUERIES:
        raise ValueError('Invalid count or lengths')
    rng = np.random.default_rng(seed)
    seen = set() if seen is None else seen
    width = max(lengths)
    x = np.zeros((count, width), dtype=np.int64)
    y = np.full_like(x, -100)
    distance = np.full_like(x, -1)
    lens = np.empty(count, dtype=np.int64)
    row = 0
    while row < count:
        length = int(rng.choice(lengths))
        keys = rng.choice(np.arange(1, 17), PAIRS, replace=False)
        values = rng.integers(17, 33, PAIRS)
        tokens = rng.integers(33, 41, length)
        tokens[:2 * PAIRS:2] = keys
        tokens[1:2 * PAIRS:2] = values
        which = rng.choice(PAIRS, QUERIES, replace=False)
        positions = rng.choice(np.arange(2 * PAIRS, length), QUERIES, replace=False)
        tokens[positions] = keys[which]
        # Reject shared memory assignments, even when queries/fillers differ.
        identity = tuple(sorted(zip(keys.tolist(), values.tolist())))
        if identity in seen:
            continue
        seen.add(identity)
        x[row, :length] = tokens
        y[row, positions] = values[which]
        distance[row, positions] = positions - (2 * which + 1)
        lens[row] = length
        row += 1
    return dict(zip(FIELDS, (x, y, distance, lens)))


def oracle(tokens):
    """Independent left-to-right parser; consults only input tokens, never metadata."""
    memory, locations, labels, gaps = {}, {}, [], []
    i = 0
    while i < len(tokens):
        token = int(tokens[i])
        if token == PAD:
            if any(tokens[i:]):
                raise ValueError('Padding must be a suffix')
            labels.extend([-100] * (len(tokens) - i))
            gaps.extend([-1] * (len(tokens) - i))
            break
        if 1 <= token <= 16:
            if i + 1 < len(tokens) and 17 <= tokens[i + 1] <= 32:
                if token in memory:
                    raise ValueError('Duplicate definition')
                memory[token] = int(tokens[i + 1])
                locations[token] = i + 1
                labels.extend([-100, -100])
                gaps.extend([-1, -1])
                i += 2
                continue
            labels.append(memory[token])
            gaps.append(i - locations[token])
        elif 33 <= token <= 40:
            labels.append(-100)
            gaps.append(-1)
        else:
            raise ValueError('Unexpected value token')
        i += 1
    return np.array(labels), np.array(gaps)


def validate(data, seen=None):
    seen = set() if seen is None else seen
    for row, labels, gaps, length in zip(*(data[k] for k in FIELDS)):
        assert length == np.count_nonzero(row)
        expected_y, expected_distance = oracle(row)
        np.testing.assert_array_equal(labels, expected_y)
        np.testing.assert_array_equal(gaps, expected_distance)
        assert np.count_nonzero(labels != -100) == QUERIES
        identity = tuple(sorted(zip(row[:2 * PAIRS:2].tolist(), row[1:2 * PAIRS:2].tolist())))
        assert identity not in seen, 'Memory assignment overlap'
        seen.add(identity)
    return seen


def datasets(c, seed, include_test=True):
    seen = set()
    specs = [('train', c['train_size'], c['train_lengths']),
             ('validation', c['validation_size'], c['train_lengths'])]
    if include_test:
        specs += [(f'test{length}', c['test_size'], [length]) for length in c['test_lengths']]
    return {name: generate(seed + 10000 + 100000 * i, n, lengths, seen)
            for i, (name, n, lengths) in enumerate(specs)}


def digest(data):
    h = hashlib.sha256()
    for key in FIELDS:
        a = data[key].astype('<i8')
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def batch(data, indices):
    length = int(data['length'][indices].max())
    return (torch.from_numpy(data['x'][indices, :length].copy()),
            torch.from_numpy(data['y'][indices, :length].copy()))
