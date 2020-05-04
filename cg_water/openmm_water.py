from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *


# Parameters
temperature = 300.0*kelvin

# Create topology
pdb = PDBFile("tip3p.pdb")
forcefield = ForceField("tip3p.xml")
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
                                 nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinIntegrator(temperature, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(temperature)
simulation.minimizeEnergy()
simulation.reporters.append(DCDReporter(file="water.dcd", reportInterval=10))
simulation.reporters.append(StateDataReporter(file="water.csv", reportInterval=10,
                                              step=True, time=True,
                                              temperature=True, volume=True, density=True))
simulation.step(10000)
