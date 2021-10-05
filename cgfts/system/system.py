from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from cgfts.forcefield.forcefield_v2 import ForceField

__all__ = ['System']


class System(object):

    def __init__(self, kT=1.0, pressure=None):
        self._force_field = ForceField(kT=kT)
        self._molecules_types = OrderedDict()
        self._molecules_nums = {}
        self.kT = kT
        self.pressure = pressure

    def add_molecule_type(self, molecule_type, num=1):
        # TODO: check if molecule already exists in system
        self._molecules_types[molecule_type.name] = molecule_type
        self._molecules_nums[molecule_type.name] = num

    def set_molecule_num(self, molecule_name, num):
        # TODO: check if molecule name exists
        # TODO: check if num is a number
        self._molecules_nums[molecule_name] = num

    @property
    def molecule_types(self):
        return iter(self._molecules_types.values())

    @property
    def molecule_nums(self):
        for key in self._molecules_types.keys():
            yield self._molecules_nums[key]

    @property
    def kT(self):
        return self._kT

    @kT.setter
    def kT(self, value):
        try:
            self._kT = float(value)
            self._force_field.kT = self._kT
        except ValueError:
            raise ValueError("must be able to convert value of kT to 'float'")

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        try:
            if value is None:
                self._pressure = value
            else:
                self._pressure = float(value)
        except ValueError:
            raise ValueError("must be able to convert value of pressure to 'float'")

    @property
    def force_field(self):
        return self._force_field

    @force_field.setter
    def force_field(self, value):
        if not isinstance(value, ForceField):
            raise TypeError("'force_field' attribute must be of type ForceField, not {}".format(type(value).__name__))
        self._force_field = value

    def to_sim(self):
        import sim

        # initialize sim_System
        sim_World = sim.chem.World(list(self._molecules_types.values()), Dim=3, Units=sim.units.DimensionlessUnits)
        sim_System = sim.system.System(sim_World, Name="system")

        # add molecules to sim_System
        for key, val in self._molecules_nums.items():
            for _ in val:
                sim_System += self._molecules_types[key].to_sim().New()

        # TODO: set system box length

        # TODO: set system temperature and pressure

        # add force field to sim_System
        sim_System.ForceField.extend(self._force_field.to_sim())

        # set up the histograms for potentials
        for P in sim_System.ForceField:
            P.Arg.SetupHist(NBin=10000, ReportNBin=100)
        sim_System.World.GetBondOrdMatrix(ShowRigid=True)

        # lock and load
        print("loading system")
        sim_System.Load()

        # initial positions and velocities
        print("initializing positions and velocities")
        sim.system.positions.CubicLattice(sim_System)
        sim.system.velocities.Canonical(sim_System, Temp=self._temperature)

        # configure integrator
        print("configuring integrator")
        Int = sim_System.Int
        Int.Method = Int.Methods.VVIntegrate
        Int.Method.TimeStep = self._time_step
        Int.Method.Thermostat = Int.Method.ThermostatLangevin
        Int.Method.LangevinGamma = 1.0
        # if ensemble == 'npt':
        #     Int.Method.Barostat = Int.Method.BarostatMonteCarlo

        # # create optimizer
        # Map = sim.atommap.PosMap()
        # for i, a in enumerate(Sys.Atom):
        #     Map.append(sim.atommap.AtomMap(Atoms1=i, Atom2=a))
        # Opt = sim.srel.optimizetrajomm.OptimizeTrajOpenMMClass(Sys, Map,
        #                                                        Traj=MappedTrj, FilePrefix="{}{}".format(Sys.Name, system_index),
        #                                                        TempFileDir=os.getcwd(),
        #                                                        UseTarHists=False)
        # Opt.ConstrainNeutralCharge()
        # self._optimizers.append(Opt)
        # self._systems.append(Sys)
        # system_index += 1
