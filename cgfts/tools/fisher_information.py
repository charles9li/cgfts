from __future__ import absolute_import, print_function

import mdtraj as md
import numba
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

    def compute(self, atom_name_1, atom_name_2=None, smear_length=0.5, save_files=True):
        topology = self._traj.topology
        atom_1_indices = np.array([a.index for a in topology.atoms_by_name(atom_name_1)])
        if atom_name_2 is None or atom_name_1 == atom_name_2:
            atom_name_2 = atom_name_1
            atom_2_indices = atom_1_indices
            same_bead_type = True
        else:
            atom_2_indices = np.array([a.index for a in topology.atoms_by_name(atom_name_2)])
            same_bead_type = False
        structure_function = _compute_helper(self._traj.xyz, self._traj.unitcell_lengths,
                                             atom_1_indices, atom_2_indices, smear_length, same_bead_type)

        if save_files:
            np.save("structure_function_{}_{}".format(atom_name_1, atom_name_2), structure_function)
            self._create_statistics_file(structure_function, atom_name_1, atom_name_2)

        return structure_function

    @staticmethod
    def _create_statistics_file(structure_function, atom_name_1, atom_name_2):
        from pymbar import timeseries
        from scipy.stats import chi2

        # total degrees of freedom
        n_total = len(structure_function)

        # find uncorrelated frames
        indices = timeseries.subsampleCorrelatedData(structure_function)
        sf_n = structure_function[indices]
        n_uncorrelated = len(indices)
        dof_uncorrelated = n_uncorrelated - 1

        # variances
        variance_unbiased = np.var(sf_n, ddof=1)
        variance_mle = np.var(sf_n)

        output = ""
        output += "num frames (total) : {} \n".format(n_total)
        output += "num frames (uncorrelated) : {} \n".format(n_uncorrelated)
        output += "mean : {} \n".format(np.mean(sf_n))
        output += "variance (unbiased) : {} \n".format(variance_unbiased)
        output += "variance (MLE) : {} \n".format(variance_mle)
        lower_bound = dof_uncorrelated * variance_unbiased / chi2.ppf(0.975, dof_uncorrelated)
        upper_bound = dof_uncorrelated * variance_unbiased / chi2.ppf(0.025, dof_uncorrelated)
        output += "variance bounds (95% CI): {} , {} \n".format(lower_bound, upper_bound)

        with open("fisher_info_stats_{}_{}.txt".format(atom_name_1, atom_name_2), 'w') as f:
            f.write(output)


@numba.jit(nopython=True, parallel=True)
def _compute_helper(xyz, unitcell_lengths, atom_1_indices, atom_2_indices, smear_length, same_bead_type):

    # num frames
    n_frames = xyz.shape[0]

    kappa = 1. / (4 * smear_length ** 2)
    prefactor = (kappa / np.pi) ** 1.5

    # initialize structure function
    structure_function = np.zeros(n_frames)

    a1_indices = _determine_a1_indices(atom_1_indices, same_bead_type)
    for i, a1 in enumerate(a1_indices):
        a2_indices = _determine_a2_indices(atom_1_indices, atom_2_indices, i + 1, same_bead_type)
        for j, a2 in enumerate(a2_indices):
            r1 = xyz[:, a1, :]
            r2 = xyz[:, a2, :]
            r12 = r2 - r1
            r12_min_image = r12 - unitcell_lengths * np.round_(r12 / unitcell_lengths, 0, np.empty(r12.shape))
            structure_function += np.exp(-np.sum(r12_min_image ** 2, axis=1) * kappa)

    # multiply everything by prefactor
    structure_function *= prefactor

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
