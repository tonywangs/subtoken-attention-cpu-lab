import copy
import math
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from subtoken_lab.model import Subtoken, Model
from subtoken_lab.data import arrays, digest


def reference(module, x):
    # Independent loops over batch, time, projection, query and key.
    batches = []
    for batch in x:
        times = []
        for token in batch:
            z = [F.linear(token, module.project.weight[j*module.r:(j+1)*module.r],
                          module.project.bias[j*module.r:(j+1)*module.r]) for j in range(module.k)]
            values = []
            for query in z:
                weights = torch.stack([torch.dot(query, key) / math.sqrt(module.r) for key in z]).softmax(0)
                values.append(sum(weights[j] * z[j] for j in range(module.k)))
            pooled = sum(values) / module.k
            times.append(F.linear(pooled, module.out.weight, module.out.bias))
        batches.append(torch.stack(times))
    return torch.stack(batches)


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reference_forward_and_gradients(self):
        for k in (1, 4):
            torch.manual_seed(7)
            a = Subtoken(5, 3, k).double()
            b = copy.deepcopy(a)
            x = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            actual, expected = a(x), reference(b, y)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            upstream = torch.randn_like(actual)
            (actual * upstream).sum().backward()
            (expected * upstream).sum().backward()
            for left, right in [(x.grad, y.grad)] + [(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters())]:
                self.assertTrue(torch.isfinite(left).all())
                torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)

    def test_numeric_gradcheck(self):
        torch.manual_seed(4)
        m = Subtoken(3, 2, 4).double()
        x = torch.randn(1, 2, 3, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-5))

    def test_k1_is_projection_composition(self):
        m = Subtoken(5, 3, 1)
        x = torch.randn(2, 3, 5)
        torch.testing.assert_close(m(x), m.out(m.project(x)))

    def test_token_and_batch_isolation(self):
        m = Subtoken()
        x = torch.randn(2, 4, 32, requires_grad=True)
        before = m(x)
        after = x.detach().clone()
        after[1] += 100
        after[0, 3] -= 100
        torch.testing.assert_close(before[0, :3], m(after)[0, :3], atol=0, rtol=0)
        before[0, 1].sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(x.grad[0, [0, 2, 3]]).item(), 0)

    def test_complete_model_causal_and_batch_isolation(self):
        for method in ('attention', 'mlp', 'pool'):
            torch.manual_seed(9)
            m = Model(method).eval()
            x = torch.randint(0, 19, (2, 13))
            y = x.clone()
            y[0, 7:] = (y[0, 7:] + 1) % 19
            y[1] = (y[1] + 2) % 19
            torch.testing.assert_close(m(x)[0, :7], m(y)[0, :7], atol=0, rtol=0)
            loss = m(x).square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_parameter_match(self):
        counts = [sum(p.numel() for p in Model(m).parameters()) for m in ('attention', 'mlp', 'pool')]
        self.assertLess(max(counts) / min(counts), 1.02)

    def test_dataset_disjoint_and_semantics(self):
        for task in ('copy', 'recall'):
            data = arrays(task, 123)
            again = arrays(task, 123)
            seen = set()
            for split, (x, y) in data.items():
                self.assertEqual(digest((x, y)), digest(again[split]))
                for row, target in zip(x.tolist(), y.tolist()):
                    prompt = tuple(row[:8] if task == 'copy' else row)
                    self.assertNotIn(prompt, seen)
                    seen.add(prompt)
                    if task == 'copy':
                        self.assertEqual(row[1:7], target[7:])
                        self.assertEqual(row[8:], target[7:-1])
                        self.assertEqual(target[:7], [-100]*7)
                    else:
                        mapping = dict(zip(row[1:9:2], row[2:9:2]))
                        self.assertEqual(len(mapping), 4)
                        self.assertEqual(mapping[row[-1]], target[-1])
                        self.assertEqual(target[:-1], [-100]*10)

    def test_pool_equals_average_projections(self):
        m = Subtoken(attention=False)
        x = torch.randn(2, 5, 32)
        weight = m.project.weight.reshape(4, 16, 32).mean(0)
        bias = m.project.bias.reshape(4, 16).mean(0)
        torch.testing.assert_close(m(x), m.out(F.linear(x, weight, bias)))


if __name__ == '__main__':
    unittest.main()
