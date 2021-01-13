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

    @classmethod
    def from_dcd(cls, dcd, top, chain_index=0):
        trajectory = md.load_dcd(dcd, top)
        return cls(trajectory, chain_index=chain_index)

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

    @property
    def rg(self):
        return self._rg

    def add_bead_type_mass(self, bead_type, mass):
        self._bead_type_mass[bead_type] = mass

    def compute(self):

        # determine bead masses
        if self._bead_type_mass:
            bead_masses = np.array([self._bead_type_mass[bead_type] for bead_type in self._bead_types])
        else:
            bead_masses = np.ones(self._n_beads)

        # compute rg
        self._rg = md.compute_rg(self._trajectory, masses=bead_masses)

    def save_to_csv(self, filename="rg.csv"):
        data = {"frame": np.arange(len(self._rg)) + 1,
                "rg (nm)": self._rg}
        df = pd.DataFrame(data=data)
        df.to_csv(path_or_buf=filename, index=False)
