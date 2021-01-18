from __future__ import absolute_import

import sys
import unittest

from cgfts.simwrapper.system import *


class TestCG(unittest.TestCase):

    def test_plma(self):
        sys.setrecursionlimit(2000)
        system = SystemCG(293.15, cut=3.0)
        self.assertEqual(4500.0, system.pressure)
        system.add_trajectory("data/50mA12_NPT_293K_1bar_3wt.dcd", "data/50mA12_NPT_293K_1bar_3wt.pdb", stride=1)
        system.add_residue_map('mA1', ['Bplma', 'C6', 'E6'], [6, 6, 6],
                               [14.027, 12.011, 15.035, 12.011] + 2*[15.999] + 11*[14.027] + [15.035])
        system.add_residue_map('C12', ['D', 'D'], [6, 6], [15.035] + 10*[14.027] + [15.035])
        system.fix_bonded('D', 'D')
        system.create_system()
        for s in system.systems:
            for p in s.ForceField:
                if p.Name == "Bonded_D_D":
                    self.assertEqual(True, p.Param.Dist0.Fixed)
                    self.assertEqual(True, p.Param.FConst.Fixed)


if __name__ == '__main__':
    unittest.main()
