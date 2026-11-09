"""CPU-only structural/logic tests; not a new scientific evaluation."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from codec import make_mapping
from decision import add_counts, empty
from scan import map_bank, patch_bank, score_maps


class CoreTests(unittest.TestCase):
    def test_source_hashes(self):
        root = Path(__file__).parent
        manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
        for item in manifest['sources']:
            actual = hashlib.sha256((root / item['output']).read_bytes()).hexdigest()
            self.assertEqual(actual, item['output_sha256'])

    def test_mapping_inverse_and_vectorization(self):
        z = torch.arange(4 * 64 * 64).reshape(1, 4, 64, 64)
        for seed in range(5):
            mapping = make_mapping(16, 5, seed)
            native = mapping.shuffler().shuffle(z)
            fast = map_bank(patch_bank(z), [mapping.position_indices], [mapping.transform_indices])
            self.assertTrue(torch.equal(native, fast))
            self.assertTrue(torch.equal(mapping.shuffler().unshuffle(native), z))

    def test_batch_scores(self):
        class Model:
            def decoder(self, z):
                return z.flatten(1)[:, :64]
        z = torch.linspace(0, 1, 4 * 64 * 64).reshape(1, 4, 64, 64)
        mappings = [make_mapping(16, 5, i) for i in range(7)]
        p = [m.position_indices for m in mappings]
        t = [m.transform_indices for m in mappings]
        message = torch.zeros(1, 64)
        a, _ = score_maps(Model(), z, p, t, message, 1)
        b, _ = score_maps(Model(), z, p, t, message, 4)
        np.testing.assert_array_equal(a, b)

    def test_exact_prefix_and_tie_rejection(self):
        state = empty()
        add_counts(state, np.array([57, 20, 58], dtype=np.uint8), 0, [2, 4, 6])
        add_counts(state, np.array([58, 59, 12], dtype=np.uint8), 3, [2, 4, 6])
        self.assertEqual([p['max_bits'] for p in state['pools']], [57, 58, 59])
        self.assertEqual([p['tie_count'] for p in state['pools']], [1, 2, 1])
        self.assertEqual([p['reject'] for p in state['pools']], [0, 1, 0])
        self.assertEqual([sum(p['histogram']) for p in state['pools']], [2, 4, 6])

    def test_incomplete_order_is_rejected(self):
        with self.assertRaises(AssertionError):
            add_counts(empty(), np.array([57, 56], dtype=np.uint8), 1, [2])


if __name__ == '__main__':
    unittest.main()
