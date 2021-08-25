from __future__ import absolute_import, print_function

import mdtraj as md
import numba
import numpy as np


class FisherInformation(object):

    def __init__(self, traj, structure_function=None):
        self._traj = traj
        self._structure_function = structure_function

    @classmethod
    def from_dcd(cls, dcd, top, stride=1):
        trajectory = md.load_dcd(dcd, top=top, stride=stride)
        return cls(trajectory)

    @property
    def traj(self):
        return self._traj

    @property
    def structure_function(self):
        return self._structure_function

    @structure_function.setter
    def structure_function(self, value):
        self._structure_function = value

    def compute(self, atom_name_1, atom_name_2=None):
        topology = self._traj.topology
        atom_1_indices = np.array([a.index for a in topology.atoms_by_name(atom_name_1)])
        if atom_name_2 is None or atom_name_1 == atom_name_2:
            return self._compute_helper_same(self._traj.xyz, atom_1_indices)
        else:
            atom_2_indices = np.array([a.index for a in topology.atoms_by_name(atom_name_2)])
            return self._compute_helper_different(self._traj.xyz, atom_1_indices, atom_2_indices)

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _compute_helper_different(xyz, atom_1_indices, atom_2_indices):

        smear_length = 0.5  # TODO: make this changeable

        structure_function = np.zeros(xyz.shape[0])

        for i, a1 in enumerate(atom_1_indices):
            for a2 in atom_2_indices:
                if a1 != a2:
                    atom_1_positions = xyz[:, a1, :]
                    atom_2_positions = xyz[:, a2, :]
                    r2 = np.sum((atom_1_positions - atom_2_positions) ** 2, axis=1)
                    structure_function += np.exp(-r2 / (4 * smear_length ** 2)) / (4 * np.pi * smear_length ** 2) ** 1.5

        # compute structure function
        return np.var(structure_function)

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _compute_helper_same(xyz, atom_1_indices):

        smear_length = 0.5  # TODO: make this changeable

        structure_function = np.zeros(xyz.shape[0])

        for i, a1 in enumerate(atom_1_indices):
            for a2 in atom_1_indices[i:]:
                if a1 != a2:
                    atom_1_positions = xyz[:, a1, :]
                    atom_2_positions = xyz[:, a2, :]
                    r2 = np.sum((atom_1_positions - atom_2_positions) ** 2, axis=1)
                    structure_function += np.exp(-r2 / (4 * smear_length ** 2)) / (4 * np.pi * smear_length ** 2) ** 1.5

        # compute structure function
        return np.var(structure_function)
