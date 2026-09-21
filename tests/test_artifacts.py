import json
import tempfile
import unittest
from pathlib import Path
from subtoken_lab.cli import sha, verify


class Artifacts(unittest.TestCase):
    def test_corrupt_artifact_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / 'config.json'
            artifact.write_text('{}')
            (root / 'manifest.json').write_text(json.dumps({'config.json': sha(artifact)}))
            artifact.write_text('{"tampered":true}')
            with self.assertRaisesRegex(AssertionError, 'config.json'):
                verify(root)

    def test_unlisted_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{}')
            (root / 'unexpected').write_text('extra')
            with self.assertRaises(AssertionError):
                verify(root)
