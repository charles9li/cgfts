import mdtraj as md
import numpy as np


class MapTrajectory(object):

    def __init__(self, trajectory):
        self._trajectory = trajectory
        self._mapped_trajectory = None
        self._residue_map = {}

    @classmethod
    def from_dcd(cls, dcd, top, stride=1):
        if isinstance(dcd, str):
            dcd = [dcd]
        trajectory_list = []
        for d in dcd:
            trajectory_list.append(md.load_dcd(d, top, stride=stride))
        return cls(md.join(trajectory_list))

    @property
    def mapped_trajectory(self):
        return self._mapped_trajectory

    def slice(self, chain_indices=0):
        if isinstance(chain_indices, int):
            chain_indices = [chain_indices]
        atom_indices = []
        for chain in self._trajectory.topology.chains:
            if chain.index in chain_indices:
                for atom in chain.atoms:
                    atom_indices.append(atom.index)
        self._trajectory = self._trajectory.atom_slice(atom_indices)

    def add_residue_map(self, residue_name, bead_names, num_atoms_per_bead, atom_masses):
        if len(bead_names) != len(num_atoms_per_bead):
            raise ValueError("bead_names must be same length as num_atoms_per_bead")
        if np.sum(num_atoms_per_bead) != len(atom_masses):
            raise ValueError("sum of num_atoms must be same as length of atom_masses")
        self._residue_map[residue_name] = (bead_names, num_atoms_per_bead, atom_masses)

    def get_n_beads(self):
        n_beads = 0
        for residue in self._trajectory.topology.residues:
            n_beads += len(self._residue_map[residue.name][0])
        return n_beads

    def map(self):

        # initialize new topology and mapped positions
        topology_mapped = md.Topology()
        xyz_mapped = np.empty((self._trajectory.n_frames, self.get_n_beads(), 3))

        # create new topology and com positions
        bead_dict = {}
        num_bead_types = 0
        bead_index = 0
        for chain in self._trajectory.topology.chains:

            # add mapped chain to new topology
            chain_mapped = topology_mapped.add_chain()

            # add residues to new topology
            for residue in chain.residues:

                # add residue to new topology
                residue_mapped = topology_mapped.add_residue(residue.name, chain_mapped)

                # get atom indices
                atom_indices = [atom.index for atom in residue.atoms]

                # get residue information
                bead_names, num_atoms_per_bead, atom_masses = self._residue_map[residue.name]

                # split atom indices and masses
                atom_indices_split = np.split(np.array(atom_indices), np.cumsum(num_atoms_per_bead))
                atom_masses_split = np.split(np.array(atom_masses), np.cumsum(num_atoms_per_bead))

                # map residue to new topology
                for indices, masses, bead_name in zip(atom_indices_split, atom_masses_split, bead_names):

                    # create element if not exists
                    if bead_name not in bead_dict.keys():
                        bead_dict[bead_name] = md.element.Element(200+num_bead_types, bead_name,
                                                                  'A'+str(num_bead_types), np.sum(masses), 1.0)
                        num_bead_types += 1

                    # add bead to new topology
                    topology_mapped.add_atom(bead_name, bead_dict[bead_name], residue_mapped)

                    # slice trajectory
                    traj_slice = self._trajectory.atom_slice(indices)

                    # compute mapped positions
                    xyz_mapped[:, bead_index, :] = np.average(traj_slice.xyz, axis=1, weights=masses)
                    bead_index += 1

        # save mapped trajectory
        self._mapped_trajectory = md.Trajectory(xyz_mapped, topology_mapped,
                                                unitcell_lengths=self._trajectory.unitcell_lengths,
                                                unitcell_angles=self._trajectory.unitcell_angles)

    def save_mapped_trajectory(self, dcd_filename, pdb_filename):
        self._mapped_trajectory.save(dcd_filename)
        self._mapped_trajectory[0].save(pdb_filename)
