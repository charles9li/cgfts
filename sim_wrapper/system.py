from collections import OrderedDict
import os

from scipy.constants import N_A, R
import mdtraj as md
import numpy as np

import sim

from sim_wrapper.utils import *

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
_TEMPERATURE_CONVERSION_FACTOR = 1.0e-25 * N_A
_PRESSURE_CONVERSION_FACTOR = R / 1.0e3


class BaseSystem(object):

    def __init__(self, temperature, pressure, cut=2.5, *args, **kwargs):
        self._system = None
        self.temperature = temperature
        self.pressure = pressure
        self._cut = cut
        self._bead_type_dict = {}
        self._mol_type_dict = OrderedDict()
        self._mol_num_dict = OrderedDict()
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

    def run(self, *args, **kwargs):
        pass


class SystemCG(BaseSystem):

    def __init__(self, temperature, pressure, cut=2.5):
        super(SystemCG, self).__init__(temperature, pressure, cut=cut)
        self._traj_list = []
        self._residue_map = {}
        self._systems = []
        self._optimizers = []

    def add_trajectory(self, dcd_list, top, stride=1):
        if isinstance(dcd_list, str):
            dcd_list = [dcd_list]
        traj_list = [md.load_dcd(dcd, top=top, stride=stride) for dcd in dcd_list]
        traj_union = md.join(traj_list)
        traj_union.xyz = traj_union.xyz / 10.
        traj_union.unitcell_lengths = traj_union.unitcell_lengths / 10.
        self._traj_list.append(traj_union)

    def add_residue_map(self, residue_name, bead_name_list, num_atoms_per_bead, atom_masses):
        self._residue_map[residue_name] = (bead_name_list, num_atoms_per_bead, atom_masses)

    def create_system(self):
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
                    if np.sum(num_atoms_per_bead) != len(masses_residue):
                        raise ValueError("Sum of atoms per bead list does not match the "
                                         "number of heavy atom masses in residue {}.".format(residue.name))

                    # create maps
                    indices_split = np.split(heavy_atom_indices, np.cumsum(num_atoms_per_bead[:-1]))
                    masses_split = np.split(masses_residue, np.cumsum(num_atoms_per_bead[:-1]))
                    for bead_type, indices, masses in zip(bead_types, indices_split, masses_split):
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
            BoxL = traj.unitcell_lengths[0][0]
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
                    Label = "Gaussian_{}{}".format(BeadName1_sorted, BeadName2_sorted)
                    BeadType1 = CG_atom_type_dict[BeadName1_sorted]
                    BeadType2 = CG_atom_type_dict[BeadName2_sorted]
                    Filter = sim.atomselect.PolyFilter([BeadType1, BeadType2])
                    Gaussian = sim.potential.LJGaussian(Sys, Cut=self._cut, Filter=Filter,
                                                        B=1.0, Kappa=1.0, Dist0=0.0, Sigma=1.0, Epsilon=0.0,
                                                        Label=Label)
                    # TODO: implement way to fix B
                    Gaussian.Param.Kappa.Fixed = True
                    Gaussian.Param.Dist0.Fixed = True
                    Gaussian.Param.Sigma.Fixed = True
                    Gaussian.Param.Epsilon.Fixed = True
                    ForceField.append(Gaussian)

            # create Bonded potentials
            print("creating bonded potentials")
            for BondType in BondTypes:
                BeadName1, BeadName2 = BondType
                Label = "Bonded_{}{}".format(BeadName1, BeadName2)
                BeadType1 = CG_atom_type_dict[BeadName1]
                BeadType2 = CG_atom_type_dict[BeadName2]
                Filter = sim.atomselect.PolyFilter([BeadType1, BeadType2], Bonded=True)
                Bonded = sim.potential.Bond(Sys, Filter=Filter,
                                            Dist0=1.0, FConst=1.0,
                                            Label=Label)
                # TODO: implement way to fix dist0 and k
                ForceField.append(Bonded)

            # add potentials to forcefield
            Sys.ForceField.extend(ForceField)

            # set up the histograms
            for P in Sys.ForceField:
                P.Arg.SetupHist(NBin=10000, ReportNBin=100)

            Sys.World.GetBondOrdMatrix(ShowRigid=True)

            # lock and load
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
            Int.Method.TimeStep = 0.005
            Int.Method.Thermostat = Int.Method.ThermostatLangevin
            Int.Method.LangevinGamma = 1.0
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
            system.ForceField.SetParamString(open(filename, 'r').read())

    def run(self, steps_equil, steps_prod, steps_stride):
        for opt in self._optimizers:
            opt.StepsEquil = steps_equil
            opt.StepsProd = steps_prod
            opt.StepsStride = steps_stride
        weights = [1.]*len(self._optimizers)
        optimizer = sim.srel.OptimizeMultiTrajClass(self._optimizers, Weights=weights)
        optimizer.FilePrefix = ("system")
        optimizer.RunConjugateGradient(MaxIter=None, SteepestIter=0)


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
        integrator.Method.TimeStep = 0.005
        integrator.Method.Thermostat = integrator.Method.ThermostatLangevin
        integrator.Method.LangevinGamma = 1.0
        integrator.Method.Barostat = integrator.Method.BarostatMonteCarlo

    def run(self, steps_equil, steps_prod, write_freq):
        sim.export.omm.MakeOpenMMTraj(self._system, DelTempFiles=False, Prefix="system_",
                                      TrajFile="traj.dcd", Verbose=True,
                                      NStepsEquil=steps_equil, NStepsProd=steps_prod,
                                      WriteFreq=write_freq)
