from __future__ import absolute_import, division, print_function

from pymbar import timeseries
from scipy.stats import linregress
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

from .com_traj import COMTraj
from .unwrap import unwrap_traj


class ComputeDiffusion(object):

    def __init__(self, traj, dt=0.01):
        self._traj = traj
        self._residue_masses = {}
        self._dt = dt
        self._msd_data = []
        self._result = None

    @classmethod
    def from_dcd(cls, dcd, top, stride=1, dt=0.01):
        if isinstance(dcd, str):
            dcd = [dcd]
        traj_list = []
        for d in dcd:
            traj_list.append(md.load_dcd(d, top=top, stride=stride))
        trajectory = md.join(traj_list)
        return cls(trajectory, dt=dt)

    def unwrap(self, method='npt'):
        self._traj = unwrap_traj(self._traj, method=method)

    def com(self, method='centroid'):
        com_traj = COMTraj(self._traj)
        for key, val in self._residue_masses.items():
            com_traj.add_residue_masses(key, val)
        com_traj.compute_com(method=method)
        self._traj = com_traj.traj_com

    def add_residue_masses(self, residue_name, masses):
        self._residue_masses[residue_name] = np.array(masses)
        
    def _compute_msd(self, tau):
        xyz = self._traj.xyz
        delta = int(tau / self._dt)
        sd = np.sum((xyz[delta:] - xyz[:-delta])**2, axis=2)
        msd = np.mean(sd, axis=1)
        indices = timeseries.subsampleCorrelatedData(msd)
        msd_n = msd[indices]
        # msd_n = msd
        self._msd_data.append((tau, np.mean(msd_n), np.std(msd_n), len(msd_n)))

    def compute_msd(self, tau):
        try:
            iter(tau)
        except TypeError:
            tau = [tau]
        for t in tau:
            self._compute_msd(t)

    def linreg(self):
        data = np.array(self._msd_data)
        self._result = linregress(data[:, 0], data[:, 1])

    @property
    def D(self):
        return self._result.slope / 6

    @property
    def slope(self):
        return self._result.slope

    @property
    def intercept(self):
        return self._result.intercept

    def plot(self):
        data = np.array(self._msd_data)
        plt.figure()
        plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2] / data[:, 3], fmt='o')
        tau_range = np.array([0, np.max(data[:, 0])])
        plt.plot(tau_range, self._result.slope*tau_range + self._result.intercept)
        plt.xlabel(r"$\tau$ (ns)")
        plt.ylabel("mean square displacement (nm^2)")

    def plot_resid(self):
        data = np.array(self._msd_data)
        plt.figure()
        resid = data[:, 1] - (self.slope*data[:, 0] + self.intercept)
        plt.scatter(data[:, 0], resid)
        tau_range = np.array([0, np.max(data[:, 0])])
        plt.plot(tau_range, [0, 0])
        plt.xlabel(r"$\tau$ (ns)")
        plt.ylabel("msd resid")

    def print_summary(self, verbose=True, summary_filename=None):

        # create summary
        s = "\nDiffusion coefficient summary"
        s += "\n============================="
        s += "\nD = {} nm^2/ns".format(self.D)
        s += "\n============================="
        s += "\ntau (ns)\tmsd (nm^2)\t\tstd dev (nm^2)\tn indep frames"
        s += "\n--------\t----------\t\t--------------\t--------------"
        for row in self._msd_data:
            s += "\n{}\t\t\t{}\t{}\t{}".format(*row)

        # print if verbose
        if verbose:
            print(s)

        # save to filename if specified
        if summary_filename is not None:
            print(s, file=open(summary_filename, 'w'))
