from scipy.constants import N_A, R
import numpy as np

import sim

__all__ = ['SystemRun']


_TEMPERATURE_CONVERSION_FACTOR = 1.0e-25 * N_A
_PRESSURE_CONVERSION_FACTOR = R / 1.0e3


class BaseSystem(object):

    def __init__(self, temperature, pressure, cut=2.5, *args, **kwargs):
        self._system = None
        self.temperature = temperature
        self.pressure = pressure
        self._cut = cut
        self._bead_type_dict = {}
        self._mol_type_dict = {}
        self._mol_num_dict = {}
        self._bond_types = []

    @property
    def system(self):
        return self._system

    @property
    def temperature(self):
        return self._temperature / _PRESSURE_CONVERSION_FACTOR

    @temperature.setter
    def temperature(self, value):
        self._temperature = value * _PRESSURE_CONVERSION_FACTOR

    @property
    def pressure(self):
        return self._pressure / _TEMPERATURE_CONVERSION_FACTOR

    @pressure.setter
    def pressure(self, value):
        self._pressure = value * _TEMPERATURE_CONVERSION_FACTOR

    @property
    def cut(self):
        return self._cut

    def create_system(self, *args, **kwargs):
        pass
    
    def load_params_from_file(self, filename):
        self._system.ForceField.SetParamString(open(filename, 'r').read())


class SystemCG(BaseSystem):

    def __init__(self, AA_traj, AA_top, temperature, pressure, cut=2.5, stride=1):
        super(SystemCG, self).__init__(temperature, pressure, cut=cut)
        raise NotImplementedError("the {} class has not been implemented".format(type(self).__name__))


_OPERATOR_PRECEDENCE = {'+': 2,
                        '*': 3}
_OPERATORS = list(_OPERATOR_PRECEDENCE.keys())
_DELIMITERS = ['(', ')']


def tokenize_expr(expr):
    token_list = expr.split()
    for char in _OPERATORS + _DELIMITERS:
        temp_list = []
        for token in token_list:
            token = [t.strip() for t in token.split(char)]
            token_split = [char] * (len(token) * 2 - 1)
            token_split[0::2] = token
            temp_list += token_split
        token_list = temp_list
    return [t for t in token_list if t]


def create_stack(token_list):
    output = []
    stack = []
    for token in token_list:
        if token in _OPERATORS:
            while len(stack) > 0 and stack[-1] != '(' and _OPERATOR_PRECEDENCE[stack[-1]] >= _OPERATOR_PRECEDENCE[token]:
                output.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            output.append(token)
    while len(stack) > 0:
        output.append(stack.pop())
    return output


def evaluate_output(output):
    stack = []
    for item in output:
        if item in _OPERATORS:
            op2 = stack.pop()
            op1 = stack.pop()
            if item == '+':
                stack.append(op1 + op2)
            elif item == '*':
                stack.append(op1 * op2)
        else:
            try:
                stack.append(int(item))
            except ValueError:
                stack.append([item])
    if len(stack) != 1:
        raise ValueError("Malformed expression.")
    return stack.pop()


class SystemRun(BaseSystem):

    def __init__(self, temperature, pressure, cut=2.5):
        super(SystemRun, self).__init__(temperature, pressure, cut=cut)

    def add_dodecane_2bead(self, num_mol=1):
        bead_name_list = ['D', 'D']
        mol_type = self._create_mol_type("dodecane", bead_name_list)
        mol_type.Bond(0, 1)
        self._bond_types.append(tuple(['D', 'D']))
        self._mol_type_dict[tuple(bead_name_list)] = mol_type
        self._mol_num_dict[tuple(bead_name_list)] = num_mol
        
    _MONOMER_TO_BEAD_NAME = {'A4': ['Bpba', 'C4'],
                             'A12': ['Bpla', 'C6', 'E6'],
                             'mA12': ['Bplma', 'C6', 'E6']}

    def add_polyacrylate(self, sequence, num_mol=1):
        monomer_list = evaluate_output(create_stack(tokenize_expr(sequence)))
        bead_name_list = []
        for monomer in monomer_list:
            bead_name_list.extend(self._MONOMER_TO_BEAD_NAME[monomer])
        mol_type = self._create_mol_type(sequence, bead_name_list)
        prev = None
        prev_bead_name = None
        curr = 0
        curr_bead_name = None
        for i, monomer in enumerate(monomer_list):
            beads_in_monomer = self._MONOMER_TO_BEAD_NAME[monomer]
            curr_bead_name = beads_in_monomer[0]
            n_beads = len(beads_in_monomer)
            for j in range(n_beads - 1):
                bead_index_1 = curr + j
                bead_index_2 = bead_index_1 + 1
                mol_type.Bond(bead_index_1, bead_index_2)
                bead_name_1 = beads_in_monomer[j]
                bead_name_2 = beads_in_monomer[j+1]
                bond_type = tuple(np.sort([bead_name_1, bead_name_2]))
                if bond_type not in self._bond_types:
                    self._bond_types.append(bond_type)
            if prev is not None:
                mol_type.Bond(prev, curr)
                bond_type = tuple(np.sort([prev_bead_name, curr_bead_name]))
                if bond_type not in self._bond_types:
                    self._bond_types.append(bond_type)
            prev_bead_name = curr_bead_name
            prev = curr
            curr += n_beads
        self._mol_type_dict[tuple(monomer_list)] = mol_type
        self._mol_num_dict[tuple(monomer_list)] = num_mol

    def _create_mol_type(self, name, bead_name_list):
        for bead_name in bead_name_list:
            if bead_name not in self._bead_type_dict.keys():
                self._bead_type_dict[bead_name] = sim.chem.AtomType(bead_name)
        return sim.chem.MolType(name, [self._bead_type_dict[bead_name] for bead_name in bead_name_list])

    def create_system(self, box_length):

        # create world
        world = sim.chem.World(self._mol_type_dict.values(), Dim=3, Units=sim.units.DimensionlessUnits)

        # create system
        self._system = sim.system.System(world, Name="system")
        for mol_type, mol_num in zip(self._mol_type_dict.values(), self._mol_num_dict.values()):
            for _ in range(mol_num):
                self._system += mol_type.New()

        # set system box length, temperature, and pressure
        self._system.BoxL = box_length
        self._system.TempSet = self._temperature
        self._system.PresSet = self._pressure
        self._system.ForceField.Globals.Charge.Fixed = True

        # initialize force field
        force_field = []

        # add Gaussian potentials
        bead_name_list = self._bead_type_dict.keys()
        for i, bead_name_1 in enumerate(bead_name_list):
            for bead_name_2 in bead_name_list[i:]:
                bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
                label = "Gaussian_{}_{}".format(bead_name_1, bead_name_2)
                bead_type_1 = self._bead_type_dict[bead_name_1]
                bead_type_2 = self._bead_type_dict[bead_name_2]
                poly_filter = sim.atomselect.PolyFilter([bead_type_1, bead_type_2])
                gaussian = sim.potential.LJGaussian(self._system, Cut=self._cut, Filter=poly_filter,
                                                    B=1.0, Kappa=1.0, Dist0=0.0, Sigma=1.0, Epsilon=0.0,
                                                    Label=label)
                # TODO: add potential fix
                force_field.append(gaussian)

        # add bonded potentials
        for bond_type in self._bond_types:
            bead_name_1, bead_name_2 = bond_type
            label = "Bonded_{}_{}".format(bead_name_1, bead_name_2)
            bead_type_1 = self._bead_type_dict[bead_name_1]
            bead_type_2 = self._bead_type_dict[bead_name_2]
            poly_filter = sim.atomselect.PolyFilter([bead_type_1, bead_type_2], Bonded=True)
            bonded = sim.potential.Bond(self._system, Filter=poly_filter,
                                        Dist0=1.0, FConst=1.0,
                                        Label=label)
            # TODO: add potential fix
            force_field.append(bonded)

        # add potentials to force field
        self._system.ForceField.extend(force_field)

        # set up the histograms
        for potential in self._system.ForceField:
            potential.Arg.SetupHist(NBin=10000, ReportNBin=100)

        # lock and load
        self._system.Load()

        # initialize positions and velocities
        sim.system.positions.CubicLattice(self._system)
        sim.system.velocities.Canonical(self._system, Temp=self._temperature)

        # configure integrator
        integrator = self._system.Int
        integrator.Method = integrator.Methods.VVIntegrate
        integrator.Method.TimeStep = 0.0005
        integrator.Method.Thermostat = integrator.Method.ThermostatLangevin
        integrator.Method.LangevinGamma = 1.0
        integrator.Method.Barostat = integrator.Method.BarostatMonteCarlo

    def run(self, steps_equil, steps_prod, write_freq):
        sim.export.omm.MakeOpenMMTraj(self._system, DelTempFiles=False, Prefix="system_",
                                      TrajFile="traj.dcd", Verbose=True,
                                      NStepsEquil=steps_equil, NStepsProd=steps_prod,
                                      WriteFreq=write_freq)
