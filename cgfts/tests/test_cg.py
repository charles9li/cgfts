from __future__ import absolute_import

import sys
import unittest

from cgfts.simwrapper.system import *


class TestCG(unittest.TestCase):

    def test_plma(self):
        sys.setrecursionlimit(2000)
        system_plma = SystemCG(313.15, 3866.4, cut=3.0)
        dcd = "/home/charlesli/lab/shell/polyiv/tests/data/plma_50mA12_NPT_313K_3866bar_3wt_1map_traj.dcd"
        top = "/home/charlesli/lab/shell/polyiv/tests/data/plma_50mA12_NPT_313K_3866bar_3wt_1map_equilibrated.pdb"
        system_plma.add_trajectory(dcd, top, stride=10)
        system_plma.add_residue_map('mA1', ['Bplma', 'C6', 'E6'], [6, 6, 6],
                                    [14.027, 12.011, 15.035, 12.011] + 2*[15.999] + 11*[14.027] + [15.035])
        system_plma.add_residue_map('C12', ['D', 'D'], [6, 6], [15.035] + 10*[14.027] + [15.035])
        system_plma.create_system()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
