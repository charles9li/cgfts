import unittest

from cgfts.forcefield import ForceField


class TestForceField(unittest.TestCase):
    
    def test_plma(self):
        ff = ForceField.from_file(313.15, "forcefields/100mA12_5wt_NPT_313K_3866bar_ff.dat")
        self.assertEqual(4, len(ff._bead_types))
        self.assertEqual(10, len(ff._gaussian_potentials))
        self.assertEqual(4, len(ff._bonded_potentials))
        monomers, interactions = ff.to_fts()
        print(monomers)
        print(interactions)


if __name__ == '__main__':
    unittest.main()
