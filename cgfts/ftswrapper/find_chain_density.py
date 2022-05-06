from __future__ import absolute_import

import os

from scipy.optimize import brentq
import numpy as np

from .operator_data import OperatorData

__all__ = ['find_chain_density']


def find_chain_density(run_fts, target_pressure, directory="find_chain_density"):

    # store current npw and read_input_fields of run_fts
    tmp_npw = run_fts.cell.npw
    tmp_input_fields_file = run_fts.init_fields.input_fields_file
    tmp_read_input_fields = run_fts.init_fields.read_input_fields

    # running single point simulations
    run_fts.cell.npw = 1
    run_fts.init_fields.read_input_fields = False

    # find pressure
    c_chain_density_a = run_fts.composition.c_chain_density
    pressure_difference_a = _compute_pressure_difference(c_chain_density_a, target_pressure, run_fts, directory)

    # store field configuration as seed for future runs
    run_fts.init_fields.input_fields_file = "fields_k.bin"
    run_fts.init_fields.read_input_fields = True

    # opposite sign of pressure difference determines search direction
    search_dir = -np.sign(pressure_difference_a)

    # find the other end of bracketing interval for chain density
    c_chain_density_b = None
    while c_chain_density_b is None:
        c_chain_density_curr = c_chain_density_a + search_dir * 0.1
        pressure_difference_curr = _compute_pressure_difference(c_chain_density_curr, target_pressure, run_fts,
                                                                directory)
        if np.sign(pressure_difference_curr) == np.sign(pressure_difference_a):
            c_chain_density_a = c_chain_density_curr
            pressure_difference_a = pressure_difference_curr
        else:
            c_chain_density_b = c_chain_density_curr

    # use Brent's method to find optimal chain density
    c_chain_density = brentq(_compute_pressure_difference, c_chain_density_a, c_chain_density_b,
                             args=(target_pressure, run_fts, directory))
    run_fts.composition.c_chain_density = c_chain_density

    # restore npw and read_input_fields
    run_fts.cell.npw = tmp_npw
    run_fts.init_fields.input_fields_file = tmp_input_fields_file
    run_fts.init_fields.read_input_fields = tmp_read_input_fields

    # return
    return c_chain_density


def _compute_pressure_difference(c_chain_density, target_pressure, run_fts, directory):

    # set chain density
    run_fts.composition.c_chain_density = c_chain_density

    # run fts
    run_fts.run(filename="params.in", directory=directory)

    # get pressure from operators file
    operators_path = os.path.join(directory, "operators.dat")
    od = OperatorData(operators_path)
    return od.get_by_column_name('Pressure')[-1] - target_pressure
