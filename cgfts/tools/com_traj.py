import mdtraj as md
import numpy as np


class COMTraj(object):

    def __init__(self, trajectory):
        self._trajectory = trajectory
        self._trajectory_com = None
        self._residue_masses = {}

    @classmethod
    def from_dcd(cls, dcd, top, stride=1):
        trajectory = md.load_dcd(dcd, top=top, stride=stride)
        return cls(trajectory)

    def add_residue_masses(self, residue_name, masses):
        self._residue_masses[residue_name] = masses

    def compute_com(self, method='centroid'):
        # TODO: implement this
        pass
