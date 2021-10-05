from __future__ import absolute_import, division, print_function

import sim

from ._molecule import _Molecule
from cgfts.utils import SortedBinarySet


class GeneralMolecule(_Molecule):

    def __init__(self, name):
        super(GeneralMolecule, self).__init__(name)
        self._beads = []
        self._bonds = []

    @property
    def bonds(self):
        return iter(self._bonds)

    @property
    def beads(self):
        return iter(self._beads)

    def add_bead(self, bead_name):
        self._beads.append(bead_name)
        return len(self._beads) - 1

    def add_bond(self, bead_index_1, bead_index_2):

        # check arguments
        if bead_index_1 == bead_index_2:
            raise ValueError("beads cannot be bonded to themselves")
        if bead_index_1 >= len(self._beads):
            raise ValueError("bead_index_1 is out of range")
        if bead_index_2 >= len(self._beads):
            raise ValueError("bead_index_2 is out of range")

        # create bond
        bond = SortedBinarySet(bead_index_1, bead_index_2)

        # check if bond already exists
        if bond in self._bonds:
            raise ValueError("the bond between beads {} and {} already exists".format(bead_index_1, bead_index_2))

        # add bond to molecule
        self._bonds.append(bond)

    @property
    def n_beads(self):
        return len(self._beads)

    def to_sim(self, force_field):
        sim_AtomTypes = [force_field.get_bead_type(bn).to_sim() for bn in self._beads]
        sim_MolType = sim.chem.MolType(Name=self.name, AtomTypes=sim_AtomTypes)
        for b in self._bonds:
            bead_index_1, bead_index_2 = list(b)
            sim_MolType.Bond(bead_index_1, bead_index_2)
        return sim_MolType
