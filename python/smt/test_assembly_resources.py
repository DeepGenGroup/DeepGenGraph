import tempfile
import unittest
from pathlib import Path
from check_assembly_resources import inspect_assembly


class AssemblyResourcesTests(unittest.TestCase):
    def test_budgets_spills_and_encoding(self):
        entry = '''  - .args: []
    .name: kernel
    .vgpr_count: {vgpr}
    .sgpr_count: 60
    .vgpr_spill_count: {spill}
    .sgpr_spill_count: 0
    .private_segment_fixed_size: {scratch}
    .group_segment_fixed_size: {lds}
'''
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / 'kernel.s'
            for encoding in ('utf-8', 'utf-16'):
                for vgpr, spill, scratch, lds, accepted in (
                    (216, 0, 0, 16384, True), (250, 0, 0, 16384, True),
                    (251, 0, 0, 16384, False), (255, 0, 0, 16384, False),
                    (256, 0, 0, 16384, False), (256, 61, 248, 16384, False),
                    (128, 1, 4, 16384, False), (128, 0, 0, 32768, False)):
                    body = entry.format(vgpr=vgpr, spill=spill, scratch=scratch, lds=lds)
                    path.write_text('amdhsa.kernels:\n' + body + 'amdhsa.version: [1, 1]\n', encoding=encoding)
                    self.assertEqual(inspect_assembly(path)['accepted'], accepted)
                path.write_text('amdhsa.kernels:\n' + body.replace('    .vgpr_count: 128\n', ''), encoding=encoding)
                with self.assertRaises(ValueError):
                    inspect_assembly(path)
            path.write_text('amdhsa.kernels:\n' + entry.format(vgpr=128, spill=0, scratch=0, lds=8192)
                            + entry.format(vgpr=256, spill=1, scratch=4, lds=16384))
            report = inspect_assembly(path)
            self.assertEqual(len(report['kernels']), 2)
            self.assertFalse(report['accepted'])


if __name__ == '__main__':
    unittest.main()
