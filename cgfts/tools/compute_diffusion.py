from __future__ import absolute_import

from pymbar import timeseries
from scipy.stats import linregress
import mdtraj as md
import numpy as np

from .com_traj import COMTraj
from .unwrap import unwrap_traj


class ComputeDiffusion(object):

    def __init__(self, traj, dt=0.01):
        self._traj = traj
        self._residue_masses = {}
        self._msd = None
        self._dt = dt
        self._diff_coeff_list = None

    @classmethod
    def from_dcd(cls, dcd, top, stride=1, dt=0.01):
        trajectory = md.load_dcd(dcd, top=top, stride=stride)
        return cls(trajectory, dt=dt)

    @property
    def msd(self):
        return self._msd

    def add_residue_masses(self, residue_name, masses):
        self._residue_masses[residue_name] = np.array(masses)
        
    def compute_msd(self, method='npt', com=False):
        
        # compute traj com
        com_traj = COMTraj(self._traj)
        for key, val in self._residue_masses.items():
            com_traj.add_residue_masses(key, val)
        if com:
            com_traj.compute_com(method='com')
        else:
            com_traj.compute_com(method)
        traj_com = com_traj.traj_com
        
        # unwrap traj_com
        traj_com_uw = unwrap_traj(traj_com, method=method)

        # compute msd
        self._msd = md.rmsd(traj_com_uw, traj_com_uw)**2

    def compute(self, tau):

        # number of frames per block
        n_frames_per_block = int(tau / self._dt)

        # compute slope of each section
        slope_list = np.empty(len(self._msd) - n_frames_per_block)
        for i in range(len(self._msd) - n_frames_per_block):
            slope, _, _, _, _ = linregress(self._dt*np.arange(n_frames_per_block), self._msd[i:i+n_frames_per_block])
            slope_list[i] = slope

        # find independent list of slopes
        indices = timeseries.subsampleCorrelatedData(slope_list)
        slope_list_n = slope_list[indices]

        # compute diffusion coefficient
        self._diff_coeff_list = slope_list_n / 6.

    def print_summary(self):
        s = "Diffusion coefficient summary"
        s += "\nquantity\tvalue"
        s += "\nmean    \t{} nm^2/s".format(np.mean(self._diff_coeff_list))
        s += "\nstd err \t{} nm^2/s".format(np.std(self._diff_coeff_list) / len(self._diff_coeff_list))
        print(s)
