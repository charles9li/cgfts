from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *


# Create topology
pdb = PDBFile("tip3p.pdb")
forcefield = ForceField("tip3p.xml")
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
                                 nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(DCDReporter(file="water.dcd", reportInterval=10))
simulation.step(1000)
