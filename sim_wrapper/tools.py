import mdtraj as md
import numpy as np
import pandas as pd


class ComputeRg(object):

    def __init__(self, trajectory, chain_index=0):

        # slice trajectory so that only the chain of interest is kept
        self._slice_trajectory(trajectory, chain_index)

        # determine the type of each bead and number of beads
        self._determine_bead_types()
        self._n_beads = len(self._bead_types)

        # mass of each bead
        self._bead_type_mass = {}

        # initialize rg
        self._rg = None

    def _slice_trajectory(self, trajectory, chain_index):
        topology = trajectory.topology
        chain = list(topology.chains)[chain_index]
        bead_indices = [atom.index for atom in chain.atoms]
        self._trajectory = trajectory.atom_slice(bead_indices)
        self._topology = self._trajectory.topology

    def _determine_bead_types(self):
        self._bead_types = []
        for atom in self._topology.atoms:
            self._bead_types.append(atom.name)

    @classmethod
    def from_dcd(cls, dcd, top, chain_index=0):
        trajectory = md.load_dcd(dcd, top)
        return cls(trajectory, chain_index=chain_index)

    @property
    def rg(self):
        return self._rg

    def add_bead_type_mass(self, bead_type, mass):
        self._bead_type_mass[bead_type] = mass

    def compute(self):

        # get positions
        xyz = self._trajectory.xyz

        # determine bead masses
        if self._bead_type_mass:
            bead_masses = [self._bead_type_mass[bead_type] for bead_type in self._bead_types]
        else:
            bead_masses = np.ones(self._n_beads)

        # compute time series for center of mass position
        com = np.average(xyz, axis=1, weights=bead_masses)

        # compute time series for radius of gyration
        d2 = np.sum((xyz - com[:, None, :])**2, axis=2)
        self._rg = np.sqrt(np.average(d2, axis=1, weights=bead_masses))

    def save_to_csv(self, filename="rg.csv"):
        data = {"frame": np.arange(len(self._rg)) + 1,
                "rg (nm)": self._rg}
        df = pd.DataFrame(data=data)
        df.to_csv(path_or_buf=filename, index=False)
