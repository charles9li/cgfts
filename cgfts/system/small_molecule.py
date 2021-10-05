from __future__ import absolute_import, division, print_function

import sim

from ._molecule import _Molecule


class SmallMolecule(_Molecule):

    def __init__(self, bead_name):
        super(SmallMolecule, self).__init__(bead_name)

    @property
    def n_beads(self):
        return 1

    def to_sim(self, force_field):
        sim_AtomType = force_field.get_bead_type(self.name).to_sim()
        sim_MolType = sim.chem.MolType(Name=self.name, AtomTypes=[sim_AtomType])
        return sim_MolType
