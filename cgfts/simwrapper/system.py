from __future__ import absolute_import, division

from collections import defaultdict, OrderedDict
import os

from scipy.constants import N_A, R
import mdtraj as md
import numpy as np

import sim

from cgfts.forcefield import *
from cgfts.utils import *

__all__ = ['SystemCG', 'SystemRun']


# default Simulation Package Settings
sim.export.lammps.NeighOne = 8000
sim.export.lammps.UseTable2 = True
sim.export.lammps.InnerCutoff = 1.e-6
sim.export.lammps.NPairPotentialBins = 1000
sim.export.lammps.LammpsExec = 'lmp_omp'
sim.export.lammps.UseLangevin = True
sim.export.lammps.OMP_NumThread = 8
sim.export.lammps.TableInterpolationStyle = 'spline'    # More robust than spline for highly CG-ed systems
sim.srel.optimizetrajlammps.LammpsDelTempFiles = False
sim.srel.optimizetrajlammps.UseLangevin = True
sim.export.omm.platformName = 'CUDA'                    # or 'OpenCL' or 'GPU' or 'CUDA'
sim.export.omm.device = -1                              # -1 is default, let openmm choose its own platform.
sim.export.omm.NPairPotentialKnots = 500                # number of points used to spline-interpolate the potential
sim.export.omm.InnerCutoff = 0.001                      # 0.001 is default. Note that a small value is not necessary, like in the lammps export, because the omm export interpolates to zero
sim.srel.optimizetrajomm.OpenMMStepsMin = 0             # number of steps to minimize structure, 0 is default
sim.srel.optimizetrajomm.OpenMMDelTempFiles = False     # False is Default
sim.export.omm.UseTabulated = True

# conversion factors
_PRESSURE_CONVERSION_FACTOR = 1.0e-25 * N_A
_TEMPERATURE_CONVERSION_FACTOR = R / 1.0e3

# pressure corresponding to each temperature
_DEFAULT_PRESSURES = {293.15: 4500.0,
                      313.15: 3866.4,
                      343.15: 3286.1,
                      353.15: 3157.8,
                      363.15: 2972.4,
                      373.15: 2738.8,
                      423.15: 1890.3}


class BaseSystem(object):

    def __init__(self, temperature, pressure=None, cut=2.5, ensemble='npt', time_step=0.005, *args, **kwargs):
        self._system = None
        self.temperature = temperature
        if pressure is None:
            self.pressure = _DEFAULT_PRESSURES[temperature]
        else:
            self.pressure = pressure
        self._ensemble = ensemble
        self._cut = cut
        self._time_step = time_step
        self._bead_type_dict = {}
        self._mol_type_dict = OrderedDict()
        self._mol_num_dict = OrderedDict()
        self._bond_types = []
        self._external_potentials = []

    @property
    def system(self):
        return self._system

    @property
    def temperature(self):
        return self._temperature / _TEMPERATURE_CONVERSION_FACTOR

    @temperature.setter
    def temperature(self, value):
        self._temperature = value * _TEMPERATURE_CONVERSION_FACTOR

    @property
    def pressure(self):
        return self._pressure / _PRESSURE_CONVERSION_FACTOR

    @pressure.setter
    def pressure(self, value):
        self._pressure = value * _PRESSURE_CONVERSION_FACTOR

    @property
    def cut(self):
        return self._cut

    def add_sinusoidal(self, bead_type, u_const, axis=0):
        self._external_potentials.append([bead_type, u_const, axis])

    def create_system(self, *args, **kwargs):
        pass
    
    def load_params_from_file(self, filename):
        try:
            self._system.ForceField.SetParamString(open(filename, 'r').read())
        except IOError:
            forcefield_data_dir = os.path.dirname(__file__)
            ff_path = os.path.join(forcefield_data_dir, '../forcefield/data', filename)
            self._system.ForceField.SetParamString(open(ff_path, 'r').read())

    def run(self, *args, **kwargs):
        pass


class SystemCG(BaseSystem):

    def __init__(self, temperature, pressure=None, cut=2.5, ensemble='npt', time_step=0.005, spline=False):
        super(SystemCG, self).__init__(temperature, pressure=pressure, cut=cut, ensemble=ensemble, time_step=time_step)
        self._traj_list = []
        self._residue_map = {}
        self._systems = []
        self._optimizers = []
        self._potential_fix = []
        self._spline = spline

    @property
    def systems(self):
        return self._systems

    def add_trajectory(self, dcd_list, top, stride=1, t0=0):
        if isinstance(dcd_list, str):
            dcd_list = [dcd_list]
        traj_list = [md.load_dcd(dcd, top=top, stride=stride) for dcd in dcd_list]
        traj_list[0] = traj_list[0][int(t0/stride)+1:]
        traj_union = md.join(traj_list)
        traj_union.xyz = traj_union.xyz / 10.
        traj_union.unitcell_lengths = traj_union.unitcell_lengths / 10.
        self._traj_list.append(traj_union)

    def add_residue_map(self, residue_name, bead_name_list, num_atoms_per_bead, atom_masses=None):
        self._residue_map[residue_name] = (bead_name_list, num_atoms_per_bead, atom_masses)

    def create_system(self, box_length=None, ensemble='npt', load=True):
        CG_atom_type_dict = {}
        system_index = 0
        for traj in self._traj_list:

            # load topology
            topology = traj.topology

            # create map
            Map = sim.atommap.PosMap()

            # map CG bead indices to
            CG_bead_index = 0
            AA_to_CG = {}
            CG_to_AA = {}
            CG_to_chain = {}
            chain_to_CG = {}
            CGAtomNameList = []
            MolTypeList = []
            MolTypeDict = OrderedDict()
            MolNumDict = OrderedDict()

            # create H atom map
            H_atom_map = defaultdict(list)
            H_atom_indices = topology.select('element == H')
            for bond in topology.bonds:
                if bond[0].index in H_atom_indices:
                    H_atom_map[bond[1].index].append(bond[0].index)
                elif bond[1].index in H_atom_indices:
                    H_atom_map[bond[0].index].append(bond[1].index)

            for chain in topology.chains:

                bead_type_list = []
                bead_index_list = []

                for residue in chain.residues:

                    # get corresponding data from residue_map
                    try:
                        bead_types, num_atoms_per_bead, masses_residue = self._residue_map[residue.name]
                    except KeyError:
                        raise ValueError("Cannot find {} residue in residue_map provided.".format(residue.name))

                    # check that bead_types and num_atoms_per_bead are the same length
                    if len(bead_types) != len(num_atoms_per_bead):
                        raise ValueError("Bead type list and atoms per bead list "
                                         "don't have same length for {} residue".format(residue.name))

                    # get indices of heavy atoms in residue
                    heavy_atom_indices = topology.select('resid {} and element != H'.format(residue.index))

                    # check that sum of num_atoms_per_bead equals the number of heavy atoms
                    if np.sum(num_atoms_per_bead) != len(heavy_atom_indices):
                        raise ValueError("Sum of atoms per bead list does not match the "
                                         "number of heavy atoms in residue {}.".format(residue.name))

                    # check that sum of num_atoms_per_bead equals length of mass of heavy atoms
                    if masses_residue is not None:
                        if np.sum(num_atoms_per_bead) != len(masses_residue):
                            raise ValueError("Sum of atoms per bead list does not match the "
                                             "number of heavy atom masses in residue {}.".format(residue.name))
                    else:
                        masses_residue = np.zeros(np.sum(num_atoms_per_bead))

                    # create maps
                    indices_split = np.split(heavy_atom_indices, np.cumsum(num_atoms_per_bead[:-1]))
                    masses_split = np.split(masses_residue, np.cumsum(num_atoms_per_bead[:-1]))
                    for bead_type, indices, masses in zip(bead_types, indices_split, masses_split):

                        # add hydrogen indices
                        indices_new = []
                        for index in indices:
                            indices_new.append(index)
                            for H_index in H_atom_map[index]:
                                indices_new.append(H_index)
                        indices = indices_new

                        # get masses from elements if not provided
                        if np.sum(masses) == 0.0:
                            masses = []
                            for index in indices:
                                masses.append(topology.atom(index).element.mass)
                            masses = np.array(masses)

                        AtomMap = sim.atommap.AtomMap(list(indices), CG_bead_index, Mass1=masses, Atom2Name=bead_type)
                        Map.append(AtomMap)
                        if bead_type not in CG_atom_type_dict.keys():
                            CG_atom_type_dict[bead_type] = sim.chem.AtomType(bead_type)
                        for i in indices:
                            AA_to_CG[i] = CG_bead_index
                        CG_to_AA[CG_bead_index] = indices
                        CG_to_chain[CG_bead_index] = chain.index
                        bead_index_list.append(CG_bead_index)
                        bead_type_list.append(CG_atom_type_dict[bead_type])
                        CGAtomNameList.append(bead_type)
                        CG_bead_index += 1

                chain_to_CG[chain.index] = bead_index_list

                MolType = sim.chem.MolType(str(chain.index), bead_type_list)
                if tuple(bead_type_list) not in MolTypeDict.keys():
                    MolTypeDict[tuple(bead_type_list)] = MolType
                    MolNumDict[tuple(bead_type_list)] = 1
                else:
                    MolNumDict[tuple(bead_type_list)] += 1
                MolTypeList.append(MolTypeDict[tuple(bead_type_list)])

            # create mapped trajectory
            print("start mapping trajectory")
            filename = "{}.lammpstrj".format("system")
            traj.save_lammpstrj(filename)
            Trj = sim.traj.Lammps(filename)
            if box_length is None:
                BoxL = traj.unitcell_lengths[0][0]
            else:
                BoxL = box_length
            MappedTrj = sim.traj.Mapped(Trj, Map, AtomNames=CGAtomNameList, BoxL=BoxL)
            print("finish mapping trajectory")

            # add bonds to MolTypes
            print("start adding bonds")
            BondTypes = []
            for bond in topology.bonds:
                AA_index_1 = bond[0].index
                AA_index_2 = bond[1].index
                CG_index_1 = AA_to_CG[AA_index_1]
                CG_index_2 = AA_to_CG[AA_index_2]
                if CG_index_1 != CG_index_2:
                    chain_index = CG_to_chain[CG_index_1]
                    MolType = MolTypeList[chain_index]
                    bead_index_list = chain_to_CG[chain_index]
                    try:
                        MolType.Bond(bead_index_list.index(CG_index_1), bead_index_list.index(CG_index_2))
                    except ValueError:
                        pass
                    AtomType1 = CGAtomNameList[CG_index_1]
                    AtomType2 = CGAtomNameList[CG_index_2]
                    BondPair = tuple(np.sort([AtomType1, AtomType2]))
                    if BondPair not in BondTypes:
                        BondTypes.append(BondPair)
            print(BondTypes)
            print("finish adding bonds")

            # define world
            print("creating world")
            World = sim.chem.World(MolTypeDict.values(), Dim=3, Units=sim.units.DimensionlessUnits)

            # create system
            print("creating system")
            Sys = sim.system.System(World, Name="system")

            # add molecules to system
            print("adding molecules to system")
            for i, MolType in zip(MolNumDict.values(), MolTypeDict.values()):
                for _ in range(i):
                    Sys += MolType.New()

            # Set system box length
            print("seting system box length")
            Sys.BoxL = BoxL

            # Set system temperature and charges
            Sys.TempSet = self._temperature
            if ensemble == 'npt':
                Sys.PresSet = self._pressure
            Sys.ForceField.Globals.Charge.Fixed = True

            # initialize force field
            ForceField = []

            # create Gaussian potentials
            print("creating Gaussian potentials")
            BeadNameList = CG_atom_type_dict.keys()
            for i, BeadName1 in enumerate(BeadNameList):
                for BeadName2 in BeadNameList[i:]:
                    BeadName1_sorted, BeadName2_sorted = np.sort([BeadName1, BeadName2])
                    if self._spline:
                        Label = "Spline_{}_{}".format(BeadName1_sorted, BeadName2_sorted)
                        BeadType1 = CG_atom_type_dict[BeadName1_sorted]
                        BeadType2 = CG_atom_type_dict[BeadName2_sorted]
                        Filter = sim.atomselect.PolyFilter([BeadType1, BeadType2])
                        Spline = sim.potential.PairSpline(Sys, Cut=self._cut, Filter=Filter, Label=Label)
                        ForceField.append(Spline)
                    else:
                        Label = "Gaussian_{}_{}".format(BeadName1_sorted, BeadName2_sorted)
                        BeadType1 = CG_atom_type_dict[BeadName1_sorted]
                        BeadType2 = CG_atom_type_dict[BeadName2_sorted]
                        Filter = sim.atomselect.PolyFilter([BeadType1, BeadType2])
                        Gaussian = sim.potential.LJGaussian(Sys, Cut=self._cut, Filter=Filter,
                                                            B=1.0, Kappa=1.0, Dist0=0.0, Sigma=1.0, Epsilon=0.0,
                                                            Label=Label)
                        Gaussian.Param.Kappa.Fixed = True
                        Gaussian.Param.Dist0.Fixed = True
                        Gaussian.Param.Sigma.Fixed = True
                        Gaussian.Param.Epsilon.Fixed = True
                        # for fix in self._potential_fix:
                        #     if Label == fix[0]:
                        #         getattr(Gaussian.Param, fix[1]).Fixed = fix[2]
                        ForceField.append(Gaussian)

            # create Bonded potentials
            print("creating bonded potentials")
            for BondType in BondTypes:
                BeadName1, BeadName2 = BondType
                if self._spline:
                    Label = "Bonded_Spline_{}_{}".format(BeadName1, BeadName2)
                    BeadType1 = CG_atom_type_dict[BeadName1]
                    BeadType2 = CG_atom_type_dict[BeadName2]
                    Filter = sim.atomselect.PolyFilter([BeadType1, BeadType2], Bonded=True)
                    Bonded_Spline = sim.potential.PairSpline(Sys, Filter=Filter, Label=Label)
                    ForceField.append(Bonded_Spline)
                else:
                    Label = "Bonded_{}_{}".format(BeadName1, BeadName2)
                    BeadType1 = CG_atom_type_dict[BeadName1]
                    BeadType2 = CG_atom_type_dict[BeadName2]
                    Filter = sim.atomselect.PolyFilter([BeadType1, BeadType2], Bonded=True)
                    Bonded = sim.potential.Bond(Sys, Filter=Filter,
                                                Dist0=1.0, FConst=1.0,
                                                Label=Label)
                    Bonded.Param.Dist0.Fixed = False
                    Bonded.Param.FConst.Fixed = False
                    # for fix in self._potential_fix:
                    #     if Label == fix[0]:
                    #         getattr(Bonded.Param, fix[1]).Fixed = fix[2]
                    ForceField.append(Bonded)

            # add external potential
            for ext_pot in self._external_potentials:
                Filter = sim.atomselect.PolyFilter([CG_atom_type_dict[ext_pot[0]]])
                Sinusoidal = sim.potential.ExternalSinusoid(Sys, Label="sin_{}".format(ext_pot[0]), Filter=Filter,
                                                            UConst=ext_pot[1], NPeriods=1.0,
                                                            PlaneAxis=ext_pot[2])
                ForceField.append(Sinusoidal)

            # add potentials to forcefield
            Sys.ForceField.extend(ForceField)

            # set up the histograms
            for P in Sys.ForceField:
                P.Arg.SetupHist(NBin=10000, ReportNBin=100)

            Sys.World.GetBondOrdMatrix(ShowRigid=True)

            # lock and load
            if load:
                print("loading system")
                Sys.Load()

            # initial positions and velocities
            print("initializing positions and velocities")
            sim.system.positions.CubicLattice(Sys)
            sim.system.velocities.Canonical(Sys, Temp=self._temperature)

            # configure integrator
            print("configuring integrator")
            Int = Sys.Int
            Int.Method = Int.Methods.VVIntegrate
            Int.Method.TimeStep = self._time_step
            Int.Method.Thermostat = Int.Method.ThermostatLangevin
            Int.Method.LangevinGamma = 1.0
            if ensemble == 'npt':
                Int.Method.Barostat = Int.Method.BarostatMonteCarlo

            # create optimizer
            Map = sim.atommap.PosMap()
            for i, a in enumerate(Sys.Atom):
                Map.append(sim.atommap.AtomMap(Atoms1=i, Atom2=a))
            Opt = sim.srel.optimizetrajomm.OptimizeTrajOpenMMClass(Sys, Map,
                                                                   Traj=MappedTrj, FilePrefix="{}{}".format(Sys.Name, system_index),
                                                                   TempFileDir=os.getcwd(),
                                                                   UseTarHists=False)
            Opt.ConstrainNeutralCharge()
            self._optimizers.append(Opt)
            self._systems.append(Sys)
            system_index += 1

    def load_params_from_file(self, filename):
        for system in self._systems:
            try:
                system.ForceField.SetParamString(open(filename, 'r').read())
            except IOError:
                forcefield_data_dir = os.path.dirname(__file__)
                ff_path = os.path.join(forcefield_data_dir, '../forcefield/data', filename)
                system.ForceField.SetParamString(open(ff_path, 'r').read())

    def fix_parameter(self, potential_name, parameter_name, fix=True):
        # self._potential_fix.append((potential_name, parameter_name, fix))
        for s in self._systems:
            for p in s.ForceField:
                if p.Name == potential_name:
                    getattr(p.Param, parameter_name).Fixed = fix

    def set_parameter_min(self, potential_name, parameter_name, min_val=0.0):
        for s in self._systems:
            for p in s.ForceField:
                if p.Name == potential_name:
                    getattr(p.Param, parameter_name).Min = min_val

    def fix_gaussian_B(self, bead_name_1, bead_name_2):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        potential_name = "Gaussian_{}_{}".format(bead_name_1, bead_name_2)
        self.fix_parameter(potential_name, 'B')

    def fix_gaussian_B_set(self, bead_name_list):
        for i, bead_name_1 in enumerate(bead_name_list):
            for bead_name_2 in bead_name_list[i:]:
                self.fix_gaussian_B(bead_name_1, bead_name_2)

    def fix_bonded_Dist0(self, bead_name_1, bead_name_2):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        potential_name = "Bonded_{}_{}".format(bead_name_1, bead_name_2)
        self.fix_parameter(potential_name, 'Dist0')

    def fix_bonded_FConst(self, bead_name_1, bead_name_2):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        potential_name = "Bonded_{}_{}".format(bead_name_1, bead_name_2)
        self.fix_parameter(potential_name, 'FConst')
        
    def fix_bonded(self, bead_name_1, bead_name_2):
        self.fix_bonded_Dist0(bead_name_1, bead_name_2)
        self.fix_bonded_FConst(bead_name_1, bead_name_2)

    def set_gaussian_parameter(self, potential_name, parameter_name, value):
        for s in self._systems:
            for p in s.ForceField:
                if p.Name == potential_name:
                    potential_string = ">>> POTENTIAL {}\n{}".format(p.Name, p.ParamString())
                    potential = Gaussian.from_string(potential_string)
                    setattr(potential, parameter_name, value)
                    s.ForceField.SetParamString(str(potential))

    def set_gaussian_B(self, bead_name_1, bead_name_2, value):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        potential_name = "Gaussian_{}_{}".format(bead_name_1, bead_name_2)
        self.set_gaussian_parameter(potential_name, 'B', value)

    def set_gaussian_B_min(self, bead_name_1, bead_name_2, value):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        potential_name = "Gaussian_{}_{}".format(bead_name_1, bead_name_2)
        self.set_parameter_min(potential_name, 'B', value)
        
    def set_gaussian_B_min_all(self, value):
        for s in self._systems:
            for p in s.ForceField:
                if p.Name.startswith('Gaussian'):
                    self.set_parameter_min(p.Name, 'B', min_val=value)

    def set_gaussian_Kappa(self, bead_name_1, bead_name_2, value):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        potential_name = "Gaussian_{}_{}".format(bead_name_1, bead_name_2)
        self.set_gaussian_parameter(potential_name, 'Kappa', value)

    def set_default_gaussian_Kappa(self, smear_length_scale=1.0):
        for s in self._systems:
            for p in s.ForceField:
                if p.Name.startswith('Gaussian'):
                    potential_string = ">>> POTENTIAL {}\n{}".format(p.Name, p.ParamString())
                    potential = Gaussian.from_string(potential_string)
                    potential.set_default_Kappa(self.temperature, smear_length_scale=smear_length_scale)
                    s.ForceField.SetParamString(str(potential))

    def set_gaussian_Kappa_all(self, value):
        for s in self._systems:
            for p in s.ForceField:
                if p.Name.startswith('Gaussian'):
                    potential_string = ">>> POTENTIAL {}\n{}".format(p.Name, p.ParamString())
                    potential = Gaussian.from_string(potential_string)
                    potential.Kappa = value
                    s.ForceField.SetParamString(str(potential))

    def set_bonded_parameter(self, potential_name, parameter_name, value):
        for s in self._systems:
            for p in s.ForceField:
                if p.Name == potential_name:
                    potential_string = ">>> POTENTIAL {}\n{}".format(p.Name, p.ParamString())
                    potential = Bonded.from_string(potential_string)
                    setattr(potential, parameter_name, value)
                    s.ForceField.SetParamString(str(potential))

    def set_bonded_Dist0(self, bead_name_1, bead_name_2, value):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        potential_name = "Bonded_{}_{}".format(bead_name_1, bead_name_2)
        self.set_bonded_parameter(potential_name, 'Dist0', value)

    def set_bonded_FConst(self, bead_name_1, bead_name_2, value):
        bead_name_1, bead_name_2 = np.sort([bead_name_1, bead_name_2])
        potential_name = "Bonded_{}_{}".format(bead_name_1, bead_name_2)
        self.set_bonded_parameter(potential_name, 'FConst', value)

    def run(self, steps_equil, steps_prod, steps_stride, constrain_mode=0):
        for opt in self._optimizers:
            opt.StepsEquil = steps_equil
            opt.StepsProd = steps_prod
            opt.StepsStride = steps_stride
            opt.ConstrainMode = constrain_mode
        weights = [1.]*len(self._optimizers)
        optimizer = sim.srel.OptimizeMultiTrajClass(self._optimizers, Weights=weights)
        optimizer.FilePrefix = ("system")
        optimizer.ConstrainMode = constrain_mode
        optimizer.RunConjugateGradient(MaxIter=None, SteepestIter=0)


class SystemRun(BaseSystem):

    def __init__(self, temperature, pressure=None, cut=2.5, ensemble='npt', time_step=0.005,):
        super(SystemRun, self).__init__(temperature, pressure=pressure, cut=cut, ensemble=ensemble, time_step=time_step)

    def add_dodecane_1bead(self, num_mol=1):
        bead_name_list = ['D12']
        mol_type = self._create_mol_type("dodecane", bead_name_list)
        self._mol_type_dict[tuple(bead_name_list)] = mol_type
        self._mol_num_dict[tuple(bead_name_list)] = num_mol

    def add_dodecane_2bead(self, num_mol=1):
        bead_name_list = ['D6', 'D6']
        mol_type = self._create_mol_type("dodecane", bead_name_list)
        mol_type.Bond(0, 1)
        self._bond_types.append(tuple(['D6', 'D6']))
        self._mol_type_dict[tuple(bead_name_list)] = mol_type
        self._mol_num_dict[tuple(bead_name_list)] = num_mol

    def add_dodecane_3bead(self, num_mol=1):
        bead_name_list = ['D4', 'D4', 'D4']
        mol_type = self._create_mol_type("dodecane", bead_name_list)
        mol_type.Bond(0, 1)
        mol_type.Bond(1, 2)
        bond_type = tuple(['D4', 'D4'])
        if bond_type not in self._bond_types:
            self._bond_types.append(tuple(['D4', 'D4']))
        self._mol_type_dict[tuple(bead_name_list)] = mol_type
        self._mol_num_dict[tuple(bead_name_list)] = num_mol

    _MONOMER_TO_BEAD_NAME_1BEAD = {'A4': ['A4'],
                                   'A12': ['A12'],
                                   'mA12': ['mA12']}

    def add_polyacrylate_1bead(self, sequence, num_mol=1):
        monomer_list = acrylate_sequence_to_list(sequence)
        bead_name_list = []
        for monomer in monomer_list:
            bead_name_list.extend(self._MONOMER_TO_BEAD_NAME_1BEAD[monomer])
        mol_type = self._create_mol_type(sequence, bead_name_list)

        for i in range(len(monomer_list) - 1):
            mol_type.Bond(i, i+1)
            bead_name_1 = bead_name_list[i]
            bead_name_2 = bead_name_list[i+1]
            bond_type = tuple(np.sort([bead_name_1, bead_name_2]))
            if bond_type not in self._bond_types:
                self._bond_types.append(bond_type)

        self._mol_type_dict[tuple(monomer_list)] = mol_type
        self._mol_num_dict[tuple(monomer_list)] = num_mol
        
    _MONOMER_TO_BEAD_NAME = {'A4': ['Bpba', 'C4'],
                             'A12': ['Bpla', 'C6', 'E6'],
                             'mA12': ['Bplma', 'C6', 'E6']}

    def add_polyacrylate(self, sequence, num_mol=1):
        monomer_list = acrylate_sequence_to_list(sequence)
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

    _MONOMER_TO_BEAD_NAME_SREL3 = {'A4': ['Bpba', 'D4'],
                                   'A12': ['Bpla', 'D4', 'D4', 'D4'],
                                   'mA12': ['Bplma', 'D4', 'D4', 'D4']}

    def add_polyacrylate_srel3(self, sequence, num_mol=1):
        monomer_list = acrylate_sequence_to_list(sequence)
        bead_name_list = []
        for monomer in monomer_list:
            bead_name_list.extend(self._MONOMER_TO_BEAD_NAME_SREL3[monomer])
        mol_type = self._create_mol_type(sequence, bead_name_list)
        prev = None
        prev_bead_name = None
        curr = 0
        curr_bead_name = None
        for i, monomer in enumerate(monomer_list):
            beads_in_monomer = self._MONOMER_TO_BEAD_NAME_SREL3[monomer]
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

    def add_pba_partial_side_chain(self, num_backbone=5, num_side_chain=5, num_mol=1):
        bead_name_list = ['Bpba']*num_backbone + ['C4']*num_side_chain
        mol_type = self._create_mol_type("{}A4_{}C4".format(num_backbone, num_side_chain), bead_name_list)
        for i in range(num_backbone - 1):
            mol_type.Bond(i, i+1)
            bond_type = tuple(np.sort(['Bpba', 'Bpba']))
            if bond_type not in self._bond_types:
                self._bond_types.append(bond_type)
        for i in range(num_side_chain):
            mol_type.Bond(i, i+num_backbone)
            bond_type = tuple(np.sort(['Bpba', 'C4']))
            if bond_type not in self._bond_types:
                self._bond_types.append(bond_type)
        self._mol_type_dict[tuple(bead_name_list)] = mol_type
        self._mol_num_dict[tuple(bead_name_list)] = num_mol

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
        if self._ensemble == 'npt':
            self._system.PresSet = self._pressure
        self._system.ForceField.Globals.Charge.Fixed = True

        # initialize force field
        force_field = []

        # add Gaussian potentials
        bead_name_list = self._bead_type_dict.keys()
        for i, bead_name_1 in enumerate(bead_name_list):
            for bead_name_2 in bead_name_list[i:]:
                bead_name_1_sorted, bead_name_2_sorted = np.sort([bead_name_1, bead_name_2])
                label = "Gaussian_{}_{}".format(bead_name_1_sorted, bead_name_2_sorted)
                bead_type_1 = self._bead_type_dict[bead_name_1_sorted]
                bead_type_2 = self._bead_type_dict[bead_name_2_sorted]
                poly_filter = sim.atomselect.PolyFilter([bead_type_1, bead_type_2])
                gaussian = sim.potential.LJGaussian(self._system, Cut=self._cut, Filter=poly_filter,
                                                    B=1.0, Kappa=1.0, Dist0=0.0, Sigma=1.0, Epsilon=0.0,
                                                    Label=label)
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
            force_field.append(bonded)

        # add external potential
        for ext_pot in self._external_potentials:
            Filter = sim.atomselect.PolyFilter([self._bead_type_dict[ext_pot[0]]])
            Sinusoidal = sim.potential.ExternalSinusoid(self._system, Label="sin_{}".format(ext_pot[0]), Filter=Filter,
                                                        UConst=ext_pot[1], NPeriods=1.0,
                                                        PlaneAxis=ext_pot[2])
            force_field.append(Sinusoidal)

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
        integrator.Method.TimeStep = self._time_step
        integrator.Method.Thermostat = integrator.Method.ThermostatLangevin
        integrator.Method.LangevinGamma = 1.0
        if self._ensemble == 'npt':
            integrator.Method.Barostat = integrator.Method.BarostatMonteCarlo

    def run(self, steps_equil, steps_prod, write_freq, use_openmm=True):
        if use_openmm:
            sim.export.omm.MakeOpenMMTraj(self._system, DelTempFiles=False, Prefix="system_",
                                          TrajFile="traj.dcd", Verbose=True,
                                          NStepsEquil=steps_equil, NStepsProd=steps_prod,
                                          WriteFreq=write_freq)
        else:
            sim.export.lammps.MakeLammpsTraj(self._system, DelTempFiles=False, Prefix="system_",
                                             TrajFile="traj.lammpstrj", Verbose=True,
                                             NStepsEquil=steps_equil, NStepsProd=steps_prod,
                                             WriteFreq=write_freq)


if __name__ == '__main__':
    t = 313.15
    print(t)
    t = t / _TEMPERATURE_CONVERSION_FACTOR
    print(t)
    t = t * _TEMPERATURE_CONVERSION_FACTOR
    print(t)
