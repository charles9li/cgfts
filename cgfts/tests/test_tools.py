from __future__ import absolute_import

import unittest

import matplotlib.pyplot as plt
import numpy as np

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

    def test_diffusion(self):

        tau_min = 20.0
        tau_max = 40.0
        tau_step = 1.0

        # filename
        top = "data/positions.pdb"

        # use NVT algorithm
        dcd = "data/traj_NVT.dcd"
        compute_diffusion_NVT = ComputeDiffusion.from_dcd(dcd, top, dt=0.01)
        compute_diffusion_NVT.add_residue_masses('C12', [15.035] + 10*[14.027] + [15.035])
        compute_diffusion_NVT.com(method='com')
        compute_diffusion_NVT.unwrap(method='nvt')
        compute_diffusion_NVT.compute_msd(np.arange(start=tau_min, stop=tau_max, step=tau_step))
        compute_diffusion_NVT.linreg()
        compute_diffusion_NVT.plot()
        compute_diffusion_NVT.plot_resid()
        print(compute_diffusion_NVT.D, compute_diffusion_NVT.intercept)
        compute_diffusion_NVT.print_summary()

        # use NPT algorithm
        dcd = "data/traj_NPT.dcd"
        compute_diffusion_NPT = ComputeDiffusion.from_dcd(dcd, top, dt=0.01)
        compute_diffusion_NPT.add_residue_masses('C12', [15.035] + 10*[14.027] + [15.035])
        compute_diffusion_NPT.com(method='com')
        compute_diffusion_NPT.unwrap(method='npt')
        compute_diffusion_NPT.compute_msd(np.arange(start=tau_min, stop=tau_max, step=tau_step))
        compute_diffusion_NPT.linreg()
        compute_diffusion_NPT.plot()
        compute_diffusion_NPT.plot_resid()
        print(compute_diffusion_NPT.D, compute_diffusion_NPT.intercept)
        compute_diffusion_NPT.print_summary()

        # plot
        plt.show()


if __name__ == '__main__':
    unittest.main()
