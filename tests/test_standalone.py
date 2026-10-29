"""Exercise the actual CLI exports, example answers and optional comparison."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_contract import release_fixture

ROOT = Path(__file__).resolve().parents[1]


class StandaloneTests(unittest.TestCase):
    def cli(self, *args):
        result = subprocess.run([sys.executable, '-m', 'exact_sequential_inference', *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_saved_examples(self):
        for mode, name in [('active_sensing', 'active'), ('noisy_temporal', 'temporal')]:
            actual = self.cli('solve', mode, ROOT / 'examples' / f'{name}-input.json')
            expected = json.loads((ROOT / 'examples' / f'{name}-output.json').read_text())
            self.assertEqual(actual, expected)

    def test_cli_exports_filter_labels_and_future_arrivals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [release_fixture('active_sensing'), release_fixture('noisy_temporal')]
            (root / 'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            model = root / 'model.jsonl'
            result = self.cli('export', root, '--view', 'model', '--output', model)
            self.assertEqual(result['exported_views'], 9)
            views = [json.loads(s) for s in model.read_text().splitlines()]
            self.assertEqual(set(views[0]), {'instructions', 'formal_input'})
            for view in views[1:]:
                self.assertEqual(set(view), {'instructions', 'formal_input', 'source_records', 'query'})
                self.assertNotIn('query_id', view['query'])
                self.assertTrue(all(r['arrival_time'] <= view['query']['cutoff'] for r in view['source_records']))
            adaptation = root / 'adaptation.jsonl'
            self.cli('export', root, '--view', 'adaptation', '--cutoff', '7', '--output', adaptation)
            view = json.loads(adaptation.read_text())
            self.assertEqual(set(view), {'formal_input', 'cutoff', 'source_records'})
            self.assertTrue(all(r['arrival_time'] <= 7 for r in view['source_records']))
            original = model.read_bytes()
            with self.assertRaises(subprocess.CalledProcessError):
                self.cli('export', root, '--view', 'model', '--output', model)
            self.assertEqual(original, model.read_bytes())

    def test_comparison_skips_without_external_checkout(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/task8_anchor_check.py')],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'skipped')
