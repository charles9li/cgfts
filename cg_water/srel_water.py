from collections import OrderedDict as ODict
import mdtraj as md
import numpy as np
import os
import sim

# Parameters
Units = sim.units.DimensionlessUnits
AAtrajs = ["water.dcd"]
AAtops = ["tip3p.pdb"]

maxIter = None
steepestIter = 0

sysName = "water"
NMolsDicts = [[('H2O', 895)]]
MolTypesDicts = [{'H2O': ['H2O']}]

Temp = 8.314e-3*300.
dt = 0.01
IntParams = {'TimeStep': dt, 'LangevinGamma': 1/(100*dt)}

StepsEquil = 0.5/dt
StepsProd = 0.5/dt
StepsStride = 1

# Default Simulation Package Settings
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
sim.srel.optimizetrajomm.OpenMMStepsMin = 0 #number of steps to minimize structure, 0 is default
sim.srel.optimizetrajomm.OpenMMDelTempFiles = False #False is Default
sim.export.omm.UseTabulated = True

# ============== #
# Map trajectory #
# ============== #

traj = md.load_dcd("water.dcd", top="tip3p.pdb")
top = traj.topology
AAatomId = []
CGatomTypes = []
AAatomsInThisCGBead = []
for atom in list(top.atoms):
    AAatomsInThisCGBead.append(atom.index)
    if atom.name == 'H2':
        CGatomTypes.append('H2O')
        AAatomId.append(AAatomsInThisCGBead)
        AAatomsInThisCGBead = []
UniqueCGAtomTypes = np.unique(CGatomTypes)

# create mapped object
Map = sim.atommap.PosMap()
for i, CGatomType in enumerate(CGatomTypes):
    Atoms1 = AAatomId[i]
    Atom2 = i
    this_map = sim.atommap.AtomMap(Atoms1, Atom2)
    Map.append(this_map)
# traj.save_lammpstrj("water.lammpstrj")
Trj = sim.traj.Lammps("water.lammpstrj", Units=Units)
BoxL = traj.unitcell_lengths[0][0]
MappedTrj = sim.traj.Mapped(Trj, Map, AtomNames=CGatomTypes, BoxL=BoxL)

# ============= #
# Create system #
# ============= #

MolTypesDict = MolTypesDicts[0]
NMolsDict = ODict(NMolsDicts[0])

AtomTypes = {}
MolTypes = []
for AtomName in UniqueCGAtomTypes:
    AtomType = sim.chem.AtomType(AtomName, Mass=18.02, Charge=0.0)
    AtomTypes.update({AtomName: AtomType})

for MolName, NMol in NMolsDict.items():
    if NMol == 0:
        break
    AtomsInMol = []
    AtomNames = MolTypesDict[MolName]
    for AtomName in AtomNames:
        AtomsInMol.append(AtomTypes[AtomName])
    MolType = sim.chem.MolType(MolName, AtomsInMol)
    MolTypes.append(MolType)
world = sim.chem.World(MolTypes, Dim=3, Units=Units)
Sys = sim.system.System(world, Name=sysName)
Sys.BoxL = BoxL

# Add molecules to system
for i, MolType in enumerate(MolTypes):
    NMol = NMolsDict[MolType.Name]
    for j in range(NMol):
        Sys += MolType.New()
Sys.ForceField.Globals.Charge.Fixed = True

# Add forcefield
Forcefield = []
P = sim.potential.LJ(Sys, Filter=sim.atomselect.Pairs, Label="LJ", Cut=1.4, Sigma=0.3, Epsilon=1.0)
Forcefield.append(P)
Sys.ForceField.extend(Forcefield)

# Set up the histograms
for P in Sys.ForceField:
    P.Arg.SetupHist(NBin=10000, ReportNBin=100)
# lock and load
Sys.Load()

# Initial positions and velocities
sim.system.positions.CubicLattice(Sys)
sim.system.velocities.Canonical(Sys, Temp=Temp)

# Configure integrator
Int = Sys.Int
Int.Method = Int.Methods.VVIntegrate
Int.Method.TimeStep = IntParams['TimeStep']
Int.Method.Thermostat = Int.Method.ThermostatLangevin
Int.Method.LangevinGamma = IntParams['LangevinGamma']

# ================ #
# Create optimizer #
# ================ #

print(Sys.NMol)
print(Sys.NAtom)
print(Sys.NDOF)
print(Sys.Atom)

Map = sim.atommap.PosMap()
for i, a in enumerate(Sys.Atom):
    Map.append(sim.atommap.AtomMap(Atoms1=i, Atom2=a))
Opt = sim.srel.optimizetrajomm.OptimizeTrajOpenMMClass(Sys, Map, Beta=1./(Units.kB*Sys.TempSet),
                                                       Traj=MappedTrj, FilePrefix="{}".format(Sys.Name),
                                                       TempFileDir=os.getcwd(),
                                                       UseTarHists=False)
Opt.ConstrainNeutralCharge()
Opt.StepsEquil = StepsEquil
Opt.StepsProd = StepsProd
Opt.StepsStride = StepsStride

Opts = [Opt]
Weights = [1.]*len(Opts)
Optimizer = sim.srel.OptimizeMultiTrajClass(Opts, Weights=Weights)
Optimizer.FilePrefix = ("{}".format(sysName))
Optimizer.RunConjugateGradient(MaxIter=maxIter, SteepestIter=steepestIter)
