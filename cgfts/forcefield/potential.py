from __future__ import absolute_import, division, print_function

from math import pi

from ._parameter import _Parameter
from cgfts.utils import SortedBinarySet

__all__ = ['_PairPotential', 'Gaussian', 'Bonded']


class _Potential(object):

    def __init__(self, force_field=None):
        self._force_field = force_field
        self._parameters = {}

    def __getattr__(self, item):
        return self._parameters[item]

    @property
    def force_field(self):
        return self._force_field

    @force_field.setter
    def force_field(self, value):
        if value.__class__.__name__ == 'ForceField':
            self._force_field = value
        else:
            error_message = "'force_field' attribute must of type " \
                            "'ForceField', not '{}'".format(value.__class__.__name__)
            raise ValueError(error_message)

    @property
    def kT(self):
        return self._force_field.kT

    @property
    def name(self):
        return "{}".format(self.__class__.__name__)

    def to_sim_string(self):
        pass


class _PairPotential(_Potential):

    def __init__(self, bead_name_1, bead_name_2, force_field=None):
        super(_PairPotential, self).__init__(force_field=force_field)
        self._bead_names = SortedBinarySet(bead_name_1, bead_name_2)

    @property
    def bead_names(self):
        return self._bead_names

    @property
    def bead_types(self):
        return [self._force_field.get_bead_type(bead_name) for bead_name in self.bead_names]

    @property
    def name(self):
        bead_name_1, bead_name_2 = self.bead_names
        return "{}_{}_{}".format(self.__class__.__name__, bead_name_1, bead_name_2)

    def to_sim_string(self):
        s = ">>> POTENTIAL {} \n".format(self.name)
        return s


class Gaussian(_PairPotential):

    def __init__(self, bead_name_1, bead_name_2, force_field=None, excl_vol=1.0):
        super(Gaussian, self).__init__(bead_name_1, bead_name_2, force_field=force_field)
        self._parameters['excl_vol'] = _Parameter('excl_vol', excl_vol, self, False)
        self._parameters['B'] = _Parameter('B', self._B_func, self, False, use_value_func=True)
        self._parameters['Kappa'] = _Parameter('Kappa', self._Kappa_func, self, True, use_value_func=True)

    @staticmethod
    def _B_func(potential):
        excl_vol = potential.excl_vol.value
        Kappa = potential.Kappa.value
        kT = potential.kT
        return excl_vol * kT * (Kappa / pi) ** 1.5

    @staticmethod
    def _Kappa_func(potential):
        smear_length_sum = 0.0
        for bead_type in potential.bead_types:
            smear_length_sum += bead_type.smear_length ** 2
        return 0.5 / smear_length_sum

    def to_sim_string(self):
        s = ">>> POTENTIAL {} \n".format(self.name)
        s += "{'Epsilon' : 0.0000e+00 , \n"
        s += " 'B' : {} , \n".format(self.B.value)
        s += " 'Kappa' : {} , \n".format(self.Kappa.value)
        s += " 'Dist0' : 0.0000e+00 , \n"
        s += " 'Sigma' : 1.0000e+00 } \n"
        return s


class Bonded(_PairPotential):

    def __init__(self, bead_name_1, bead_name_2, force_field=None, FConst=1.0, Dist0=0.0):
        super(Bonded, self).__init__(bead_name_1, bead_name_2, force_field)
        self._parameters['FConst'] = _Parameter('FConst', FConst, self, False)
        self._parameters['Dist0'] = _Parameter('Dist0', Dist0, self, False)

    def to_sim_string(self):
        s = ">>> POTENTIAL {} \n".format(self.name)
        s += "{{'Dist0' : {} , \n".format(self.Dist0.value)
        s += " 'Fconst': {} }} \n".format(self.FConst.value)
        return s
