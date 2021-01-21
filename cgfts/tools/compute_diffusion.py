from __future__ import absolute_import

import mdtraj as md
import numpy as np

from .com_traj import COMTraj
from .unwrap import unwrap_traj


class ComputeDiffusion(object):

    def __init__(self, traj):
        self._traj = traj
        self._residue_masses = {}
        self._msd = None

    @classmethod
    def from_dcd(cls, dcd, top, stride=1):
        trajectory = md.load_dcd(dcd, top=top, stride=stride)
        return cls(trajectory)

    @property
    def msd(self):
        return self._msd

    def add_residue_masses(self, residue_name, masses):
        self._residue_masses[residue_name] = np.array(masses)
        
    def compute(self, method='npt', com=False):
        
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
