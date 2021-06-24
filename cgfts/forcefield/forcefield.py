from __future__ import division

from ast import literal_eval
import os

from scipy.constants import R
import numpy as np

__all__ = ['BeadType', 'Gaussian', 'Bonded', 'ForceField']


_SMEAR_LENGTHS = {'A4':     {313.15: 0.6070253571898435,
                             373.15: 0.6157183580257802},
                  'A12':    {313.15: 0.7614438061045977,
                             373.15: 0.7726444026435343},
                  'mA12':   {313.15: 0.7748483370378075,
                             373.15: 0.7859077900317265},
                  'Bpba':   {293.15: 0.3950973097055087,
                             313.15: 0.39353296218472217,
                             333.15: 0.39003296218472,      # linearly interpolated
                             353.15: 0.38663296218472,      # linearly interpolated
                             373.15: 0.383170058124081},
                  'Bpla':   {293.15: 0.3950973097055087,
                             313.15: 0.39353296218472217,
                             333.15: 0.39003296218472,      # linearly interpolated
                             353.15: 0.38663296218472,      # linearly interpolated
                             373.15: 0.383170058124081},
                  'Bplma':  {293.15: 0.44143311416427605,
                             313.15: 0.43912067186733483,
                             353.15: 0.43427304,
                             373.15: 0.43164103662924436,
                             423.15: 0.4223270013351243,
                             473.15: 0.400554359035373},
                  'C4':     {293.15: 0.4992116008809759,
                             313.15: 0.5024653198112052,
                             333.15: 0.5057498564469449,
                             353.15: 0.5095224805001485,
                             373.15: 0.5129911009550229},
                  'D4':     {293.15: 0.4992116008809759,
                             313.15: 0.5024653198112052,
                             333.15: 0.5057498564469449,
                             353.15: 0.5095224805001485,
                             373.15: 0.5129911009550229},
                  'C6':     {313.15: 0.575160305442338,
                             353.15: 0.58325764033,
                             373.15: 0.5872282195663296,
                             423.15: 0.5987361711132173},
                  'E6':     {313.15: 0.575160305442338,
                             353.15: 0.58325764033,
                             373.15: 0.5872282195663296,
                             423.15: 0.5987361711132173},
                  'D':      {313.15: 0.575160305442338,
                             373.15: 0.5872282195663296},
                  'D6':     {313.15: 0.575160305442338,
                             353.15: 0.58325764033,
                             373.15: 0.5872282195663296,
                             423.15: 0.5987361711132173},
                  'D12':    {313.15: 0.7246803915920862,
                             373.15: 0.739861194923907}}

# pressure corresponding to each temperature
_DEFAULT_PRESSURES = {293.15: 4500.0,
                      313.15: 3866.4,
                      353.15: 3157.8,
                      373.15: 2738.8}


class BeadType(object):

    def __init__(self, name, smear_length):
        self.name = name
        self.smear_length = smear_length

    def volume(self, temperature):
        try:
            return _SMEAR_LENGTHS[self.name][temperature] ** 3
        except KeyError:
            return self.smear_length ** 3


class _Potential(object):

    def __init__(self, bead_name_1, bead_name_2, *args, **kwargs):
        self.bead_name_1, self.bead_name_2 = np.sort([bead_name_1, bead_name_2])
        self.name = "_".join([self.__class__.__name__, self.bead_name_1, self.bead_name_2])

    @classmethod
    def from_string(cls, string):
        pass


class Gaussian(_Potential):

    def __init__(self, bead_name_1, bead_name_2, B=1.0, Kappa=1.0):
        super(Gaussian, self).__init__(bead_name_1, bead_name_2)
        self.B = B
        self.Kappa = Kappa

    @classmethod
    def from_string(cls, string):
        s = string.replace("{", "")
        s = s.replace("}", "")
        s = s.replace(",", "")
        s = s.split(">>> POTENTIAL")[1]
        name = s.split('\n')[0].strip()
        bead_name_1, bead_name_2 = name.split('_')[1:]
        data = s.split('\n')[1:]
        for line in data:
            line = line.strip()
            if line.startswith("'B'"):
                B = literal_eval(line.split(":")[1].strip())
            elif line.startswith("'Kappa'"):
                Kappa = literal_eval(line.split(":")[1].strip())
        try:
            return cls(bead_name_1, bead_name_2, B, Kappa)
        except NameError:
            raise ValueError("string is not in right form")

    def __str__(self):
        s = ">>> POTENTIAL " + self.name
        s += "\n{'Epsilon' : 0.0000e+00 ,"
        s += "\n 'B' : {} ,".format(self.B)
        s += "\n 'Kappa' : {} ,".format(self.Kappa)
        s += "\n 'Dist0' : 0.0000e+00 ,"
        s += "\n 'Sigma' : 1.0000e+00 }"
        return s

    def set_default_Kappa(self, temperature, smear_length_scale=1.0):
        smear_length_1 = _SMEAR_LENGTHS[self.bead_name_1][temperature] * smear_length_scale
        smear_length_2 = _SMEAR_LENGTHS[self.bead_name_2][temperature] * smear_length_scale
        self.Kappa = 1. / (2. * (smear_length_1**2 + smear_length_2**2))


class Bonded(_Potential):

    def __init__(self, bead_name_1, bead_name_2, Dist0=1.0, FConst=1.0e3):
        super(Bonded, self).__init__(bead_name_1, bead_name_2)
        self.Dist0 = Dist0
        self.FConst = FConst

    @classmethod
    def from_string(cls, string):
        s = string.replace("{", "")
        s = s.replace("}", "")
        s = s.replace(",", "")
        s = s.split(">>> POTENTIAL")[1]
        name = s.split('\n')[0].strip()
        bead_name_1, bead_name_2 = name.split('_')[1:]
        data = s.split('\n')[1:]
        for line in data:
            line = line.strip()
            if line.startswith("'Dist0'"):
                Dist0 = literal_eval(line.split(":")[1].strip())
            elif line.startswith("'FConst'"):
                FConst = literal_eval(line.split(":")[1].strip())
        try:
            return cls(bead_name_1, bead_name_2, Dist0, FConst)
        except NameError:
            raise ValueError("string is not in right form")

    def __str__(self):
        s = ">>> POTENTIAL " + self.name
        s += "\n{{'Dist0' : {} ,".format(self.Dist0)
        s += "\n 'FConst' : {} }}".format(self.FConst)
        return s


class ForceField(object):

    def __init__(self, temperature):
        self._bead_types = []
        self._gaussian_potentials = []
        self._bonded_potentials = []
        self.temperature = temperature

    @classmethod
    def from_file(cls, temperature,  ff_file):
        ff = cls(temperature)

        try:
            s = open(ff_file, 'r').read()
        except IOError:
            forcefield_data_dir = os.path.dirname(__file__)
            ff_path = os.path.join(forcefield_data_dir, '../forcefield/data', ff_file)
            s = open(ff_path, 'r').read()

        for p_string in s.split(">>> POTENTIAL")[1:]:
            p_string = p_string.strip()
            if p_string.startswith("Gaussian"):
                p = Gaussian.from_string(">>> POTENTIAL " + p_string)
                ff.add_gaussian_potential(p)
                if p.bead_name_1 == p.bead_name_2:
                    ff.add_bead_type(BeadType(p.bead_name_1, np.sqrt(1/(4.*p.Kappa))))
            elif p_string.startswith("Bonded"):
                p = Bonded.from_string(">>> POTENTIAL " + p_string)
                ff.add_bonded_potential(p)
            else:
                raise ValueError("potential type not valid")
        return ff

    @property
    def bonded_potentials(self):
        return self._bonded_potentials

    @property
    def gaussian_potentials(self):
        return self._gaussian_potentials

    @property
    def bead_types(self):
        return self._bead_types

    def add_bead_type(self, bead_type):
        self._bead_types.append(bead_type)

    def get_bead_type(self, bead_name):
        for bead_type in self._bead_types:
            if bead_name == bead_type.name:
                return bead_type
        raise ValueError("force field has no bead type with name '{}'".format(bead_name))

    def get_bead_volume(self, bead_name):
        return self.get_bead_type(bead_name).volume(self.temperature)

    def reorder_bead_types(self, bead_types):
        if len(self._bead_types) != len(bead_types):
            raise ValueError("argument has length {}, must have length {}".format(len(bead_types), len(self._bead_types)))
        if set([b.name for b in self._bead_types]) == set(bead_types):
            new_bead_types = []
            for b in bead_types:
                for bead_type in self._bead_types:
                    if b == bead_type.name:
                        new_bead_types.append(bead_type)
            self._bead_types = new_bead_types
        else:
            raise ValueError("supplied bead types are not the same as the forcefield")

    def add_gaussian_potential(self, gaussian_potential):
        bead_name_1 = gaussian_potential.bead_name_1
        bead_name_2 = gaussian_potential.bead_name_2
        try:
            p = self.get_gaussian_potential(bead_name_1, bead_name_2)
            p.B = gaussian_potential.B
            p.Kappa = gaussian_potential.Kappa
        except ValueError:
            self._gaussian_potentials.append(gaussian_potential)

    def get_gaussian_potential(self, bead_name_1, bead_name_2):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        for p in self._gaussian_potentials:
            if bead_name_1 == p.bead_name_1 and bead_name_2 == p.bead_name_2:
                return p
        raise ValueError("no Gaussian potential between {} and {} bead types exists".format(bead_name_1, bead_name_2))

    def add_bonded_potential(self, bonded_potential):
        bead_name_1 = bonded_potential.bead_name_1
        bead_name_2 = bonded_potential.bead_name_2
        try:
            p = self.get_bonded_potential(bead_name_1, bead_name_2)
            p.Dist0 = bonded_potential.Dist0
            p.FConst = bonded_potential.FConst
        except ValueError:
            self._bonded_potentials.append(bonded_potential)

    def get_bonded_potential(self, bead_name_1, bead_name_2):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        for p in self._bonded_potentials:
            if bead_name_1 == p.bead_name_1 and bead_name_2 == p.bead_name_2:
                return p
        raise ValueError("no bonded potential between {} and {} bead types exists".format(bead_name_1, bead_name_2))

    def create_gaussian_potentials(self):
        for i, bead_type_1 in enumerate(self._bead_types):
            for bead_type_2 in self._bead_types[i:]:
                try:
                    self.get_gaussian_potential(bead_type_1.name, bead_type_2.name)
                except ValueError:
                    kappa = 1. / (2. * (bead_type_1.smear_length**2 + bead_type_2.smear_length**2))
                    self._gaussian_potentials.append(Gaussian(bead_type_1.name, bead_type_2.name, Kappa=kappa))

    def to_sim(self):
        s = ""
        for p in self._gaussian_potentials:
            s += "\n{}".format(p)
        return s

    def to_fts(self):

        # create monomers
        monomers = "  monomers {{ # {}".format(" ".join([b.name for b in self._bead_types]))
        monomers += "\n    NSpecies = {}".format(len(self._bead_types))
        # TODO: implement Kuhn lengths
        monomers += "\n    Charge = {}".format(" ".join(["0."] * len(self._bead_types)))
        monomers += "\n    GaussSmearWidth = {}".format(" ".join([str(b.smear_length) for b in self._bead_types]))
        monomers += "\n  }"

        # create interactions
        interactions = "    interactions {"
        for i, bead_type_1 in enumerate(self._bead_types):
            for j, bead_type_2 in enumerate(self._bead_types[i:]):
                gaussian = self.get_gaussian_potential(bead_type_1.name, bead_type_2.name)
                B = gaussian.B * (np.pi / gaussian.Kappa)**(3./2.) / (R * 1.e-3 * self.temperature)
                interactions += "\n      BExclVolume{}{} = {}".format(i+1, i+j+1, B)
        interactions += "\n    }"

        return monomers, interactions


if __name__ == '__main__':
    ff = ForceField(373.15)
    ff.add_bead_type(BeadType('Bpba', _SMEAR_LENGTHS['Bpba'][ff.temperature]))
    ff.add_bead_type(BeadType('C4', _SMEAR_LENGTHS['C4'][ff.temperature]))
    ff.add_bead_type(BeadType('D', _SMEAR_LENGTHS['D'][ff.temperature]))
    ff.create_gaussian_potentials()
    print(ff.to_sim())
