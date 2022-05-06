from __future__ import absolute_import, division

import unittest

from scipy.constants import R

from cgfts.forcefield.forcefield_v2 import ForceField


class TestForceField(unittest.TestCase):

    def test_dodecane(self):
        kT = R * 313.15 / 1000
        ff = ForceField.from_sim_ff_file("../forcefields/dodecane_3bead_NPT_313K_3866bar_smear0.75_ff.dat", kT=kT)
        self.assertEqual(['D4'], list(ff.bead_names))
        gaussian = ff.get_pair_potential("Gaussian", "D4", "D4")
        self.assertAlmostEqual(2.84325396156, gaussian.excl_vol.value)
        print(ff.get_bead_type("D4").smear_length)
