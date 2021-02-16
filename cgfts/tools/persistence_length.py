from MDAnalysis.analysis import polymer
from MDAnalysis.exceptions import NoDataError
import numpy as np


class PersistenceLengthPrune(polymer.PersistenceLength):

    def __init__(self, atomgroups, prune=0, **kwargs):
        super(PersistenceLengthPrune, self).__init__(atomgroups, **kwargs)
        self.prune = prune

    def _perform_fit(self):
        """Fit the results to an exponential decay"""
        try:
            self.results
        except AttributeError:
            raise NoDataError("Use the run method first")
        self.x = np.arange(len(self.results)) * self.lb

        print(self.x)
        print(self.results)
        self.lp = polymer.fit_exponential_decay(self.x[self.prune:], self.results[self.prune:])

        self.fit = np.exp(-self.x/self.lp)

    def plot(self, ax=None):
        """Visualise the results and fit
        Parameters
        ----------
        ax : matplotlib.Axes, optional
          if provided, the graph is plotted on this axis
        Returns
        -------
        ax : the axis that the graph was plotted on
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.x, self.results, 'ro', label='Result')
        ax.plot(self.x, self.fit, label='Fit')
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'$C(x)$')
        # ax.set_xlim(0.0, 40 * self.lb)

        ax.legend(loc='best')

