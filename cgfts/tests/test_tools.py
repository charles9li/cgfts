from __future__ import absolute_import

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

    def test_diffusion(self):

        import mdtraj as md

        # filename
        top = "data/positions.pdb"

        # parameters
        tau = 1.0

        # use NVT algorithm
        dcd = "data/traj_NVT.dcd"
        traj = md.load_dcd(dcd, top=top)
        compute_diffusion_NVT = ComputeDiffusion(traj, dt=0.01)
        compute_diffusion_NVT.add_residue_masses('C12', [15.035] + 10*[14.027] + [15.035])
        compute_diffusion_NVT.compute_msd(method='nvt', com=False)
        compute_diffusion_NVT.compute(tau=tau)
        compute_diffusion_NVT.print_summary(summary_filename="msd_summary_NVT.txt")

        # use NPT algorithm
        dcd = "data/traj_NPT.dcd"
        traj = md.load_dcd(dcd, top=top)
        compute_diffusion_NPT = ComputeDiffusion(traj, dt=0.01)
        compute_diffusion_NPT.add_residue_masses('C12', [15.035] + 10*[14.027] + [15.035])
        compute_diffusion_NPT.compute_msd(method='npt', com=False)
        compute_diffusion_NPT.compute(tau=tau)
        compute_diffusion_NPT.print_summary(summary_filename="msd_summary_NPT.txt")

        # plot msd
        plt.figure()
        plt.plot(compute_diffusion_NVT.msd, label="NVT")
        plt.plot(compute_diffusion_NPT.msd, label="NPT")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
