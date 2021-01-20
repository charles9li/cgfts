import unittest

import matplotlib.pyplot as plt

from cgfts.tools import *


class TestTools(unittest.TestCase):

    def test_compute_rg(self):
        dcd = "../../polyiv/tests/data/plma_50mA12_NPT_313K_3866bar_3wt_1map_traj.dcd"
        top = "../../polyiv/tests/data/plma_50mA12_NPT_313K_3866bar_3wt_1map_equilibrated.pdb"
        compute_rg = ComputeRg.from_dcd(dcd, top)
        compute_rg.add_bead_type_mass('B', 85.082)
        compute_rg.add_bead_type_mass('C', 84.162)
        compute_rg.add_bead_type_mass('D', 85.170)
        compute_rg.compute()
        compute_rg.save_to_csv()

        plt.figure()
        plt.plot(compute_rg.rg)
        plt.show()

    def test_com(self):
        dcd = "data/dodecane_NVT_313K_1bar_L4nm_M5_cf1.dcd"
        top = "data/dodecane_313K_1bar_L4nm_M5_cf1.pdb"
        com_traj = COMTraj.from_dcd(dcd, top)
        com_traj.add_residue_masses('C12', [15.035] + 10*[14.027] + [15.035])
        com_traj.compute_com(method='com')
        self.assertEqual(170, com_traj.traj_com.topology.n_atoms)


if __name__ == '__main__':
    unittest.main()
