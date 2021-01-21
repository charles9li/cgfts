# coding=utf-8
import argparse
import mdtraj as md
import numpy as np

__all__ = ['unwrap_traj', 'unwrap_traj_from_dcd']


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unwraps trajectory.")
    parser.add_argument('traj', type=str, help="trajectory file")
    parser.add_argument('top', type=str, help="topology file")
    parser.add_argument('--method', default='npt', type=str, help="method of unwrapping")
    return parser.parse_args()


def unwrap_traj(traj_w, method='npt', output_filename=None, save_traj=False):
    """Unwraps a trajectory.

    Reference
    ---------
    von Bülow, S.; Bullerjahn, J. T.; Hummer, G. Systematic Errors in Diffusion
    Coefficients from Long-TIme Molecular Dynamics Simulations at Constant Pressure.
    J. Chem. Phys. 2020, 153 (2), 021101. https://doi.org/10.1063/5.0008316.

    Parameters
    ----------
    traj_w : md.Trajectory
        wrapped trajectory
    method : str, optional
        Which algorithm to use to unwrap trajectories. If 'nvt', the
        conventional unwrapping scheme is used, and if 'npt', the scheme
        outlined in the reference is used.
    save_traj : bool, optional
        If True, saves the unwrapped trajectory to a dcd file with '_uw'
        appended to the file prefix. For example, if traj_filename is
        'test.dcd', the unwrapped trajectory will be saved to 'test_uw.dcd'

    Returns
    -------
    traj_uw : md.Trajectory
        unwrapped Trajectory
    """

    # import wrapped trajectory
    xyz_w = traj_w.xyz

    # initialize unwrapped positions
    xyz_uw = np.empty([traj_w.n_frames, traj_w.topology.n_atoms, 3])

    # frame 0 in unwrapped trajectory is same as the wrapped trajectory's
    xyz_uw[0] = xyz_w[0]

    # determine which algorithm to use
    if method == 'nvt':
        def compute_uw_frame(i, xyz_w, xyz_uw, uc_vecs, uc_vecs_inv):
            return xyz_w[i] - np.floor((xyz_w[i] - xyz_uw[i-1]).dot(uc_vecs_inv) + np.array([0.5, 0.5, 0.5])).dot(uc_vecs)
    elif method == 'npt':
        def compute_uw_frame(i, xyz_w, xyz_uw, uc_vecs, uc_vecs_inv):
            return xyz_uw[i-1] + (xyz_w[i] - xyz_w[i-1]) - np.floor((xyz_w[i] - xyz_w[i-1]).dot(uc_vecs_inv) + np.array([0.5, 0.5, 0.5])).dot(uc_vecs)
    else:
        raise ValueError("method must be 'nvt' or 'npt'")

    # compute unwrapped positions
    for i in range(1, len(xyz_uw)):
        uc_vecs = traj_w.unitcell_vectors[i]
        uc_vecs_inv = np.linalg.inv(uc_vecs)
        xyz_uw[i] = compute_uw_frame(i, xyz_w, xyz_uw, uc_vecs, uc_vecs_inv)

    # create new unwrapped trajectory
    traj_uw = md.Trajectory(xyz_uw, traj_w.topology,
                            unitcell_lengths=traj_w.unitcell_lengths, unitcell_angles=traj_w.unitcell_angles)

    # save unwrapped trajectory to file
    if save_traj:
        if output_filename is not None:
            traj_uw.save(output_filename)
        else:
            raise ValueError("no filename supplied")

    # return unwrapped trajectory
    return traj_uw


def unwrap_traj_from_dcd(traj_filename, top_filename, method='npt', save_traj=False):
    traj_w = md.load_dcd(traj_filename, top=top_filename)
    output_filename = '.'.join(traj_filename.split('.')[:-1])
    return unwrap_traj(traj_w, method=method, output_filename=output_filename, save_traj=save_traj)


if __name__ == '__main__':
    args = parse_arguments()
    unwrap_traj_from_dcd(args.traj, args.top, method=args.method, save_traj=True)
