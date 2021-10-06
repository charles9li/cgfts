from __future__ import absolute_import, division, print_function

from ast import literal_eval
from math import pi
from functools import total_ordering
import os

import numpy as np

from .potential import _PairPotential, Bonded, Gaussian
from cgfts.utils import SortedBinarySet

__all__ = ['BeadType', 'ForceField']


@total_ordering
class BeadType(object):

    def __init__(self, name, smear_length=1.0, volume=None, charge=0.0):
        self.name = name
        self.smear_length = smear_length
        if volume is None:
            self.volume = smear_length ** 3
        else:
            self.volume = volume
        self.charge = charge
        self._sim_AtomType = None

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.name == other.name

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.name < other.name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("mame of bead must be str")
        self._name = value

    @property
    def smear_length(self):
        return self._smear_length

    @smear_length.setter
    def smear_length(self, value):
        try:
            self._smear_length = float(value)
        except ValueError:
            raise ValueError("must be able to convert smear_length of BeadType {} to float".format(self.name))

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        try:
            self._volume = float(value)
        except ValueError:
            raise ValueError("must be able to convert volume of BeadType {} to float".format(self.name))

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, value):
        try:
            self._charge = float(value)
        except ValueError:
            raise ValueError("must be able to convert charge of BeadType {} to float".format(self.name))

    def to_sim(self):
        import sim
        if self._sim_AtomType is None:
            self._sim_AtomType = sim.chem.AtomType(self.name, Charge=self.charge)
        return self._sim_AtomType


class ForceField(object):

    def __init__(self, kT=1.0):
        self.kT = kT
        self._bead_types = []
        self._potentials = []

    @classmethod
    def from_sim_ff_file(cls, filepath, kT):

        # initialize force field
        ff = cls(kT)

        # open force field file
        try:
            s = open(filepath, 'r').read()
        except IOError:
            forcefield_data_dir = os.path.dirname(__file__)
            ff_path = os.path.join(forcefield_data_dir, '../forcefield/data', filepath)
            s = open(ff_path, 'r').read()

        # TODO: find way to check that all Gaussian interactions are here
        # TODO: find way to check that smear lengths are consistent

        # separate like Gaussian interactions from all other interactions
        like_gaussian_dict = {}
        other_dict = {}
        for p_string in s.split(">>> POTENTIAL")[1:]:
            p_string = p_string.replace("{", "")
            p_string = p_string.replace("}", "")
            p_string = p_string.replace(",", "")
            p_string = p_string.strip()
            potential_name = p_string.split('\n')[0].strip()
            potential_data = p_string.split('\n')[1:]
            potential_type, bead_name_1, bead_name_2 = potential_name.split("_")
            if potential_type == "Gaussian" and bead_name_1 == bead_name_2:
                like_gaussian_dict[potential_name] = potential_data
            else:
                other_dict[potential_name] = potential_data

        # use like Gaussian interactions to determine bead types
        for potential_name, potential_data in like_gaussian_dict.items():
            bead_name = potential_name.split("_")[1].strip()
            B = cls._determine_parameter(potential_data, 'B')
            Kappa = cls._determine_parameter(potential_data, 'Kappa')
            excl_vol = B / ff.kT * (pi / Kappa) ** 1.5
            smear_length = 1. / (2 * Kappa ** 0.5)
            bead_type = BeadType(bead_name, smear_length)
            ff.add_bead_type(bead_type)
            potential = Gaussian(bead_name, bead_name, force_field=ff, excl_vol=excl_vol)
            ff.add_potential(potential)

        # add all other interactions
        for potential_name, potential_data in other_dict.items():
            potential_type, bead_name_1, bead_name_2 = potential_name.split("_")
            if potential_type == "Gaussian":
                B = cls._determine_parameter(potential_data, 'B')
                Kappa = cls._determine_parameter(potential_data, 'Kappa')
                excl_vol = B / ff.kT * (pi / Kappa) ** 1.5
                potential = Gaussian(bead_name_1, bead_name_2, force_field=ff, excl_vol=excl_vol)
                ff.add_potential(potential)
            elif potential_type == "Bonded":
                Dist0 = cls._determine_parameter(potential_data, 'Dist0')
                FConst = cls._determine_parameter(potential_data, 'FConst')
                potential = Bonded(bead_name_1, bead_name_2, force_field=ff, Dist0=Dist0, FConst=FConst)
                ff.add_potential(potential)

        return ff

    @staticmethod
    def _determine_parameter(potential_data, parameter_name):
        for line in potential_data:
            line = line.strip()
            if line.startswith("'{}'".format(parameter_name)):
                return literal_eval(line.split(":")[1].strip())

    @property
    def kT(self):
        return self._kT

    @kT.setter
    def kT(self, value):
        try:
            self._kT = float(value)
        except ValueError:
            raise ValueError("must be able to convert new value of kT to float")

    def add_bead_type(self, bead_type):
        if bead_type.name in self.bead_names:
            raise ValueError("ForceField already contains BeadType '{}'".format(bead_type.name))
        self._bead_types.append(bead_type)

    def get_bead_type(self, bead_name):
        for bt in self.bead_types:
            if bead_name == bt.name:
                return bt
        raise ValueError("ForceField does not contain BeadType '{}'".format(bead_name))

    def reorder_bead_types(self, bead_names):
        if set(bead_names) != set(self.bead_names):
            raise ValueError("Supplied bead names don't match bead names in ForceField")
        temp_bead_types = [self.get_bead_type(bn) for bn in bead_names]
        self._bead_types = temp_bead_types

    def add_potential(self, potential):
        potential.force_field = self
        self._potentials.append(potential)

    def get_pair_potential(self, potential_type, bead_name_1, bead_name_2):
        bead_names = SortedBinarySet(bead_name_1, bead_name_2)
        for p in self.potentials:
            if isinstance(p, _PairPotential):
                if bead_names == p.bead_names and potential_type == type(p).__name__:
                    return p

        # raise error message if no potential exists in force field
        err_msg = "no pair potential of type '{}' between " \
                  "bead types '{}' and '{}'".format(potential_type, bead_name_1, bead_name_2)
        raise ValueError(err_msg)

    def get_potentials_of_type(self, potential_type):
        potentials = []
        for p in self.potentials:
            if type(p).__name__ == potential_type:
                potentials.append(p)
        return potentials

    @property
    def bead_types(self):
        return iter(self._bead_types)

    @property
    def bead_names(self):
        for bt in self.bead_types:
            yield bt.name

    @property
    def potentials(self):
        return iter(self._potentials)

    def to_sim_ff_file(self):
        s = ""
        for p in self._potentials:
            s += p.to_sim_string()
        return s

    def to_sim(self):
        sim_ForceField = []
        for p in self._potentials:
            sim_ForceField.append(p.to_sim())
        return sim_ForceField

    def to_PolyFTS_monomer(self, tab="  "):
        s = tab + "monomers {{ # {} \n".format(" ".join(list(self.bead_names)))
        s += tab*2 + "NSpecies = {} \n".format(len(list(self.bead_names)))
        s += tab*2 + "KuhnLen = {} \n".format(" ".join([str(k) for k in self._determine_kuhn_lengths()]))
        s += tab*2 + "Charge = {} \n".format(" ".join([str(bt.charge) for bt in self.bead_types]))
        s += tab*2 + "GaussSmearWidth = {} \n".format(" ".join([str(bt.smear_length) for bt in self.bead_types]))
        s += tab + "} \n"
        s += "\n"
        return s

    def _determine_kuhn_lengths(self):

        # initialize array
        kuhn_lengths = np.zeros(len(list(self.bead_types)))

        # determine kuhn lengths from like-like bonded interactions if possible
        like_bead_names = []
        for i, bead_name in enumerate(self.bead_names):
            try:
                p = self.get_pair_potential("Bonded", bead_name, bead_name)
                kuhn_lengths[i] = p.Dist0.value / np.sqrt(6)
                like_bead_names.append(bead_name)
            except ValueError:
                pass

        # determine kuhn lengths for other beads
        for i, bead_name in enumerate(self.bead_names):
            if kuhn_lengths[i] == 0:
                for p in self.get_potentials_of_type("Bonded"):
                    bead_names = list(p.bead_names)
                    if bead_name in bead_names:
                        if bead_name == bead_names[0]:
                            other_bead_name = bead_names[1]
                        else:
                            other_bead_name = bead_names[0]
                        if other_bead_name not in like_bead_names:
                            kuhn_lengths[i] = p.Dist0.value / np.sqrt(6)

        # catch any stragglers
        for i, bead_name in enumerate(self.bead_names):
            if kuhn_lengths[i] == 0:
                for p in self.get_potentials_of_type("Bonded"):
                    if bead_name in list(p.bead_names):
                        kuhn_lengths[i] = p.Dist0.value / np.sqrt(6)

        return kuhn_lengths
