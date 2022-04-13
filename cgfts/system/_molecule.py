from __future__ import absolute_import, division, print_function

__all__ = ['_Molecule']


class _Molecule(object):

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def n_beads(self):
        raise NotImplementedError("'n_beads' property not implemented for base _Molecule class")

    def to_graph(self, force_field):
        pass

    def to_sim(self, force_field):
        pass

    def to_PolyFTS(self, force_field, chain_index, tab="  "):
        pass
