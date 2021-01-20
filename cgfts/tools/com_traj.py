import mdtraj as md
import numpy as np


class COMTraj(object):

    def __init__(self, traj):
        self._traj = traj
        self._traj_com = None
        self._residue_masses = {}

    @classmethod
    def from_dcd(cls, dcd, top, stride=1):
        trajectory = md.load_dcd(dcd, top=top, stride=stride)
        return cls(trajectory)

    @property
    def traj_com(self):
        return self._traj_com

    def add_residue_masses(self, residue_name, masses):
        self._residue_masses[residue_name] = np.array(masses)

    def compute_com(self, method='centroid'):

        # initialize new topology and com positions
        topology_com = md.Topology()
        xyz_com = np.empty((self._traj.n_frames, self._traj.n_chains, 3))

        # create new topology and com positions
        bead_dict = {}
        num_bead_types = 0
        chain_name = "my_chain"
        for i, chain in enumerate(self._traj.topology.chains):

            # get chain masses
            if method == 'centroid':
                masses = np.ones(len(chain.atoms))
            elif method == 'com':
                masses = np.array([])
                for residue in chain.residues:
                    masses = np.append(masses, self._residue_masses[residue.name])
            else:
                raise ValueError("invalid method")
            mass_chain = masses.sum()

            # add com bead to topology
            chain_com = topology_com.add_chain()
            residue_com = topology_com.add_residue(chain_name, chain_com)
            if chain_name not in bead_dict.keys():
                bead_dict[chain_name] = md.element.Element(200+num_bead_types, chain_name,
                                                           'A'+str(num_bead_types), mass_chain, 1.0)
                num_bead_types += 1
            topology_com.add_atom(chain_name, bead_dict[chain_name], residue_com)

            # get atom indices
            atom_indices = [atom.index for residue in chain.residues for atom in residue.atoms]

            # slice trajectory
            traj_slice = self._traj.atom_slice(atom_indices)

            # compute com of chain
            xyz_chain_com = np.average(traj_slice.xyz, axis=1, weights=masses)
            xyz_com[:, i, :] = xyz_chain_com

        # save com trajectory
        self._traj_com = md.Trajectory(xyz_com, topology_com,
                                       unitcell_lengths=self._traj.unitcell_lengths,
                                       unitcell_angles=self._traj.unitcell_angles)

    def save_traj(self, dcd_filename, pdb_filename):
        self._traj_com.save(dcd_filename)
        self._traj_com[0].save(pdb_filename)
