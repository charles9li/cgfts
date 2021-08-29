from __future__ import absolute_import, print_function

import mdtraj as md
import numba
from numba.types import float64, int64
import numpy as np


class FisherInformation(object):

    def __init__(self, traj):
        self._traj = traj

    @classmethod
    def from_dcd(cls, dcd, top, stride=1):
        trajectory = md.load_dcd(dcd, top=top, stride=stride)
        return cls(trajectory)

    @property
    def traj(self):
        return self._traj

    def compute(self, atom_name_1, atom_name_2=None):
        topology = self._traj.topology
        atom_1_indices = np.array([a.index for a in topology.atoms_by_name(atom_name_1)])
        if atom_name_2 is None or atom_name_1 == atom_name_2:
            atom_name_2 = atom_name_1
            atom_2_indices = atom_1_indices
            same_bead_type = True
        else:
            atom_2_indices = np.array([a.index for a in topology.atoms_by_name(atom_name_2)])
            same_bead_type = False
        structure_function = _compute_helper(self._traj.xyz, self._traj.unitcell_vectors,
                                             atom_1_indices, atom_2_indices, same_bead_type)
        np.save("structure_function_{}_{}".format(atom_name_1, atom_name_2), structure_function)
        return structure_function


@numba.jit(nopython=True, parallel=True)
def _compute_helper(xyz, unitcell_vectors, atom_1_indices, atom_2_indices, same_bead_type):

    # set smear length
    smear_length = 0.5  # TODO: make this specifiable
    kappa = 1. / (4 * smear_length ** 2)

    # initialize structure function
    structure_function = np.zeros(xyz.shape[0])

    for i in range(len(structure_function)):
        xyz_i = xyz[i]
        uc_vecs = unitcell_vectors[i]
        uc_vecs_inv = np.linalg.inv(uc_vecs)
        a1_indices = _determine_a1_indices(atom_1_indices, same_bead_type)
        for j, a1 in enumerate(a1_indices):
            a2_indices = _determine_a2_indices(atom_1_indices, atom_2_indices, j + 1, same_bead_type)
            for k, a2 in enumerate(a2_indices):
                r1 = xyz_i[a1]
                r2 = xyz_i[a2]
                r12 = r2 - r1
                shift = np.round_(np.sum(uc_vecs_inv * r12, axis=1), 0, np.empty(3))
                r12_min_image = r12 - np.sum(uc_vecs * shift, axis=1)
                structure_function[i] += np.exp(- np.sum(r12_min_image ** 2) * kappa) * (kappa / np.pi) ** 1.5

    return structure_function


@numba.jit(nopython=True)
def _determine_a1_indices(atom_1_indices, same_bead_type):
    if same_bead_type:
        return atom_1_indices[:-1]
    else:
        return atom_1_indices


@numba.jit(nopython=True)
def _determine_a2_indices(atom_1_indices, atom_2_indices, starting_index, same_bead_type):
    if same_bead_type:
        return atom_1_indices[starting_index:]
    else:
        return atom_2_indices
