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
        system.add_residue_map('C12', ['D6', 'D6'], [6, 6], [15.035] + 10*[14.027] + [15.035])
        system.create_system()
        system.fix_bonded('D', 'D')
        for s in system.systems:
            for p in s.ForceField:
                if p.Name == "Bonded_D_D":
                    self.assertEqual(True, p.Param.Dist0.Fixed)
                    self.assertEqual(True, p.Param.FConst.Fixed)
        system.load_params_from_file("50mA12_NPT_373K_2739bar_10wt_ff.dat")
        for s in system.systems:
            for p in s.ForceField:
                if p.Name == "Gaussian_D6_D6":
                    self.assertEqual(1.2785, p.Param.B)

    def test_plma_2(self):
        sys.setrecursionlimit(2000)
        system = SystemCG(313.15, cut=3.0)
        system.add_trajectory("data/50mA12_NPT_293K_1bar_3wt.dcd", "data/50mA12_NPT_293K_1bar_3wt.pdb")
        system.add_residue_map('mA1', ['mA12'], [18],
                               [14.027, 12.011, 15.035, 12.011] + 2*[15.999] + 11*[14.027] + [15.035])
        system.add_residue_map('C12', ['D12'], [12], [15.035] + 10*[14.027] + [15.035])
        system.create_system()
        system.set_default_gaussian_Kappa()
        system.set_gaussian_B('D12', 'D12', 3.86)
        system.set_gaussian_B('D12', 'mA12', 3.86)
        system.set_gaussian_B('mA12', 'mA12', 3.86)
        system.set_bonded_Dist0('mA12', 'mA12', 0.60)
        system.set_bonded_FConst('mA12', 'mA12', 500.0)
        system.run(100, 100, 1)


if __name__ == '__main__':
    unittest.main()
