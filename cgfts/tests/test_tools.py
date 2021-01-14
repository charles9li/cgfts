import unittest

import matplotlib.pyplot as plt

from sim_wrapper.tools import *


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


if __name__ == '__main__':
    unittest.main()
