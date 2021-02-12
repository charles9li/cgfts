from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np

from cgfts.forcefield import ForceField
from cgfts.utils import *


class FTS(object):

    def __init__(self, temperature):
        super(FTS, self).__init__()

        # forcefield and system
        self._force_field = ForceField(temperature)
        self._mol_num_dict = OrderedDict()
        self._num_dodecane_2bead = 0

        # fts system parameters
        self._input_file_version = 3
        self._num_models = 1
        self._model_type = "MOLECULAR"

        # file formatting
        self._tab = "  "

    @classmethod
    def from_file(cls, temperature, ff_file):
        fts = cls(temperature)
        fts._force_field = ForceField.from_file(temperature, ff_file)
        return fts

    @property
    def force_field(self):
        return self._force_field

    def add_polyacrylate(self, sequence, num_mol=1):
        self._mol_num_dict[sequence] = num_mol

    def add_dodecane_2bead(self, num_mol=1):
        self._num_dodecane_2bead = num_mol

    def create_input_file(self, filepath="params.in"):

        # file header
        s = "InputFileVersion = {}".format(self._input_file_version)
        s += "\n"
        s += "\n"

        # add sections
        s += self.models_string()

        print(s, file=open(filepath, 'w'))

    def models_string(self):

        # section header
        s = "models {"
        s += "\n"
        s += self._tab + "NumModels = 1"
        s += "\n"
        s += self._tab + "ModelType = {}".format(self._model_type)
        s += "\n"
        s += "\n"

        # create monomers
        bead_names = [b.name for b in self._force_field.bead_types]
        smear_lengths = [b.smear_length for b in self._force_field.bead_types]
        kuhn_lengths = self._determine_kuhn_lengths()
        s += self._tab + "monomers {{ # {}".format(" ".join(bead_names))
        s += "\n"
        s += self._tab*2 + "NSpecies = {}".format(len(bead_names))
        s += "\n"
        s += self._tab*2 + "KuhnLen = {}".format(" ".join([str(k) for k in kuhn_lengths]))
        s += "\n"
        s += self._tab*2 + "Charge = " + "0. "*len(bead_names)
        s += "\n"
        s += self._tab*2 + "GaussSmearWidth = {}".format(" ".join([str(s) for s in smear_lengths]))
        s += "\n"
        s += self._tab + "}"
        s += "\n"
        s += "\n"

        # create chains
        s += self._tab + "chains {"
        s += "\n"
        n_chains = len(self._mol_num_dict.keys())
        if self._num_dodecane_2bead > 0:
            n_chains += 1
        s += self._tab*2 + "NChains = {}".format(n_chains)
        s += "\n"
        s += self._tab*2 + "PolymerReferenceN = 1"
        s += "\n"
        s += "\n"

        # TODO: create method that can make chain section for each chain

        s += self._tab + "}"
        s += "\n"
        s += "\n"

        # TODO: create model section

        # close bracket
        s += "}"

        return s

    def _determine_kuhn_lengths(self):

        # initialize array
        kuhn_lengths = np.zeros(len(self._force_field.bead_types))

        # determine kuhn lengths from like-like bonded interactions if possible
        like_bead_names = []
        for i, bead_type in enumerate(self._force_field.bead_types):
            try:
                p = self._force_field.get_bonded_potential(bead_type.name, bead_type.name)
                kuhn_lengths[i] = p.Dist0 / np.sqrt(6)
                like_bead_names.append(bead_type.name)
            except ValueError:
                pass

        # determine kuhn lengths for other beads
        for i, bead_type in enumerate(self._force_field.bead_types):
            if kuhn_lengths[i] == 0:
                for p in self._force_field.bonded_potentials:
                    if bead_type.name in [p.bead_name_1, p.bead_name_2]:
                        if bead_type.name == p.bead_name_1:
                            other_bead_name = p.bead_name_2
                        else:
                            other_bead_name = p.bead_name_1
                        if other_bead_name not in like_bead_names:
                            kuhn_lengths[i] = p.Dist0 / np.sqrt(6)

        return kuhn_lengths


if __name__ == '__main__':
    fts = FTS.from_file(313.15, "50mA12_NPT_313K_3866bar_10wt_ff.dat")
    fts.force_field.reorder_bead_types(['Bplma', 'C6', 'E6', 'D6'])
    print(fts.models_string())
