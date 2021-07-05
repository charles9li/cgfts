from __future__ import absolute_import, division, print_function

from scipy.optimize import fsolve
import numpy as np

__all__ = ['LogNormal', 'LogNormal2']


def _check_if_n_is_int(n):
    if not isinstance(n, int):
        raise TypeError("'n' must be an 'int', not '{}'".format(type(n)))


def _discrete_moment(a, n, distribution):
    _check_if_n_is_int(n)
    a = np.array(a)
    weights = distribution.pdf(a)
    return np.average(a ** n, weights=weights)


def _equations(p, distribution):
    return_equations = []
    for i in range(len(p)):
        moment = distribution.moment(i)
        discrete_moment = _discrete_moment(p, i, distribution)
        if np.isclose(moment, 0.0):
            return_equations.append(moment - discrete_moment)
        else:
            return_equations.append((moment - discrete_moment) / moment)
    return return_equations


class _Distribution(object):

    _PARAMETERS = {}

    def __init__(self):
        pass

    def __getattr__(self, item):
        return self._PARAMETERS[item]

    def pdf(self, x):
        pass

    def moment(self, n):
        raise NotImplementedError("'moment' method for base _Distribution class not implemented")

    @property
    def mean(self):
        return self.moment(1)

    @property
    def var(self):
        return self.moment(2) - self.moment(1) ** 2

    @property
    def std(self):
        return np.sqrt(self.var)

    def discrete_points(self, n):
        _check_if_n_is_int(n)
        x0 = self._discrete_points_guess(n)
        return fsolve(_equations, x0, args=self)

    def _discrete_points_guess(self, n):
        pass


class LogNormal(_Distribution):

    def __init__(self, mu=0.0, sigma=1.0):
        super(LogNormal, self).__init__()
        self._PARAMETERS['mu'] = mu
        self._PARAMETERS['sigma'] = sigma

    def pdf(self, x):
        prefactor = 1 / (x * self.sigma * np.sqrt(2 * np.pi))
        return prefactor * np.exp(- (np.log(x) - self.mu) ** 2 / (2 * self.sigma ** 2))

    def moment(self, n):
        _check_if_n_is_int(n)
        return np.exp(n * self.mu + n ** 2 * self.sigma**2 / 2)

    def discrete_points(self, n):
        _check_if_n_is_int(n)
        if n == 1:
            return [self.mean]
        moment1 = self.mean
        moment2 = self.moment(2)

        def equations(p):
            start, dx = p
            a = np.linspace(start, start + (n - 1) * dx, num=n)
            return [(moment1 - _discrete_moment(a, 1, self)) / moment1,
                    (moment2 - _discrete_moment(a, 2, self)) / moment2]

        x0 = np.array([self.mean / n, 2 / n * self.std])
        sol = fsolve(equations, x0=x0)
        return np.linspace(sol[0], sol[0] + (n - 1) * sol[1], num=n)


class LogNormal2(_Distribution):

    def __init__(self, mu=0.0, sigma=1.0):
        super(LogNormal2, self).__init__()
        self._PARAMETERS['mu'] = mu
        self._PARAMETERS['sigma'] = sigma

    def pdf(self, x):
        prefactor = np.exp(self.mu - self.sigma ** 2 / 2) / (x ** 2 * self.sigma * np.sqrt(2 * np.pi))
        return prefactor * np.exp(- (np.log(x) - self.mu) ** 2 / (2 * self.sigma ** 2))

    def moment(self, n):
        _check_if_n_is_int(n)
        return np.exp(n * self.mu + n * (n - 2) * self.sigma ** 2 / 2)

    def discrete_points(self, n):
        _check_if_n_is_int(n)
        if n == 1:
            return [self.mean]
        moment1 = self.mean
        moment2 = self.moment(2)

        def equations(p):
            start, dx = p
            a = np.linspace(start, start + (n - 1) * dx, num=n)
            return [(moment1 - _discrete_moment(a, 1, self)) / moment1,
                    (moment2 - _discrete_moment(a, 2, self)) / moment2]

        x0 = np.array([self.mean / n, 2 / n * self.std])
        sol = fsolve(equations, x0=x0)
        return np.linspace(sol[0], sol[0] + (n - 1) * sol[1], num=n)
