import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from subtoken_lab.mqar_data import generate, oracle, validate, datasets, digest, batch
from subtoken_lab.mqar_model import RecallModel, NonlinearPool, METHODS
from subtoken_lab.mqar import base_config, evaluate, state_hash, verify


class MQAR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_oracle_fixture(self):
        # Two definitions, distractor, two queries, suffix padding.
        x = [3, 18, 7, 25, 39, 7, 40, 3, 0, 0]
        y, distance = oracle(x)
        self.assertEqual(y.tolist(), [-100]*5 + [25, -100, 18, -100, -100])
        self.assertEqual(distance.tolist(), [-1]*5 + [2, -1, 6, -1, -1])
        with self.assertRaises(ValueError):
            oracle([3, 18, 0, 3])

    def test_generation_oracle_disjoint_deterministic(self):
        c = base_config()
        c.update(train_size=128, validation_size=64, test_size=64)
        data = datasets(c, 120)
        again = datasets(c, 120)
        seen = set()
        for split, shard in data.items():
            validate(shard, seen)
            self.assertEqual(digest(shard), digest(again[split]))
        with self.assertRaises(AssertionError):
            validate(data['train'], seen)
        shard = copy.deepcopy(data['train'])
        row, col = np.argwhere(shard['y'] != -100)[0]
        shard['y'][row, col] += 1
        with self.assertRaises(AssertionError):
            validate(shard)
        self.assertEqual(set(data['train']['length']), {32, 40, 48})
        self.assertGreater(len(set(np.where(data['train']['y'] != -100)[1])), 25)
        self.assertGreater(int(data['test96']['distance'].max()), 64)

    def test_batch_trimming(self):
        data = generate(1, 100, [32, 40, 48])
        ids = np.flatnonzero(data['length'] == 32)[:3]
        x, y = batch(data, ids)
        self.assertEqual(x.shape, (3, 32))
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(int((y != -100).sum()), 12)

    def test_causal_padding_batch_and_gradients(self):
        data = generate(99, 10, [32, 48])
        for method in METHODS:
            torch.manual_seed(9)
            model = RecallModel(method).eval()
            x, y = batch(data, slice(None))
            logits = model(x)
            changed = x.clone()
            changed[0, 20:] = 35
            changed[1:] = 36
            torch.testing.assert_close(logits[0, :20], model(changed)[0, :20], atol=0, rtol=0)
            for row, length in enumerate(data['length']):
                solo = model(x[row:row+1, :length])
                torch.testing.assert_close(logits[row, :length], solo[0], atol=2e-6, rtol=2e-6)
            # Explicit interior and suffix key masks: masked content must not affect later unmasked outputs.
            tokens = x[:1, :32].clone()
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            mask[:, 3:6] = True
            mask[:, 25:] = True
            modified = tokens.clone()
            modified[mask] = 38
            a, b = model(tokens, mask), model(modified, mask)
            torch.testing.assert_close(a[~mask], b[~mask], atol=0, rtol=0)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(torch.count_nonzero(model.embed.weight.grad[0]).item(), 0)

    def test_nonlinear_reference_forward_gradients(self):
        torch.manual_seed(34)
        a, x = NonlinearPool(5, 3, 4).double(), torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
        b, y = copy.deepcopy(a), x.detach().clone().requires_grad_(True)
        # Separate affine maps and erf GELU expression rather than packed reshape/GELU.
        parts = []
        for j in range(4):
            z = F.linear(y, b.project.weight[3*j:3*(j+1)], b.project.bias[3*j:3*(j+1)])
            parts.append(0.5 * z * (1 + torch.erf(z / np.sqrt(2))))
        expected = F.linear(sum(parts) / 4, b.out.weight, b.out.bias)
        actual = a(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        for left, right in [(x.grad, y.grad)] + [(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters())]:
            torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)
        self.assertTrue(torch.autograd.gradcheck(a, (x.detach().requires_grad_(True),)))

    def test_parameter_and_shared_initialization_match(self):
        for d in (32, 48):
            models = []
            for method in METHODS:
                torch.manual_seed(7)
                models.append(RecallModel(method, d))
            counts = [sum(p.numel() for p in m.parameters()) for m in models]
            self.assertLess(max(counts) / min(counts), 1.02)
            self.assertEqual(len({state_hash(m, shared=True) for m in models}), 1)
            # Attention and both pooling controls have exactly the same initial projections.
            for name, value in models[1].state_dict().items():
                if name.startswith('mix.'):
                    for model in models[2:]:
                        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)

    def test_metrics_independent_fixture(self):
        data = generate(15, 8, [32])
        targets = data['y'].copy()
        predictions = np.where(targets == -100, 33, targets)
        positions = np.argwhere(targets != -100)
        for row, col in positions[:3]:
            predictions[row, col] = 40
        class Fixed(torch.nn.Module):
            def forward(self, x):
                return F.one_hot(torch.from_numpy(predictions), 41).float() * 10
        metrics, _ = evaluate(Fixed(), data)
        self.assertEqual(metrics['accuracy'], 29/32)
        self.assertEqual(metrics['exact_match'], 7/8)
        self.assertEqual(sum(v['count'] for v in metrics['by_distance'].values()), 32)
        self.assertEqual(sum(v['correct'] for v in metrics['by_distance'].values()), 29)

    def test_artifact_integrity_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'manifest.json').write_text(json.dumps({'config.json': 'wrong'}))
            (out / 'config.json').write_text('invalid JSON, never parsed')
            with self.assertRaises(AssertionError):
                verify(out)
            (out / 'extra').write_text('unexpected')
            with self.assertRaises(AssertionError):
                verify(out)


if __name__ == '__main__':
    unittest.main()
