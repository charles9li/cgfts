import sys
import unittest

from cgfts.simwrapper.system import *


class TestRun(unittest.TestCase):

    def test_plma(self):
        sys.setrecursionlimit(2000)
        system_plma = SystemRun(313.15, 3866.4, cut=3.0)
        system_plma.add_polyacrylate("100*mA12")
        system_plma.add_dodecane_2bead(num_mol=2868)
        system_plma.create_system(10.0)
        system_plma.load_params_from_file("forcefields/100mA12_5wt_NPT_313K_3866bar_ff.dat")
        self.assertEqual(2869, len(system_plma.system.Mol))
        for potential in system_plma.system.ForceField:
            if potential.Label == "Gaussian_Bplma_Bplma":
                self.assertEqual(1.1384, potential.Param.B)
                self.assertEqual(1.2965, potential.Param.Kappa)
        system_plma.run(2, 2, 1)

    def test_pba_forcefield(self):
        system = SystemRun(313.15, cut=4.0)
        system.add_polyacrylate("50*A4")
        system.add_dodecane_2bead(num_mol=2868)
        system.create_system(10.0)
        system.load_params_from_file("50A4_NPT_313K_3866bar_10wt_ff.dat")
        for p in system._system.ForceField:
            print(p.Name)
            print(p.ParamString())


if __name__ == '__main__':
    unittest.main()
