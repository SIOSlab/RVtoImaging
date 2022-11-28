import json
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
from tqdm import tqdm

from EXOSIMS.util.get_module import get_module_from_specs
from RVtools.universe import Planet, Star, System, Universe


def create_universe(universe_params):
    script_path = Path(universe_params["script"])
    with open(script_path) as f:
        specs = json.loads(f.read())
    SU = get_module_from_specs(specs, "SimulatedUniverse")(**specs)
    # SU.__init__(SU, **specs)
    universe = ExosimsUniverse(SU, universe_params)
    return universe


class ExosimsUniverse(Universe):
    """
    Class for the whole EXOSIMS universe
    """

    def __init__(self, SU, params):
        """
        Args:
            path (str or Path):
                Location of all the system files. Should be something like "data/1/"
        """
        # if cache:
        #     cache_base = Path(".cache", path.split("/")[1])
        #     if not cache_base.exists():
        #         cache_base.mkdir(parents=True)
        # self.path = path
        self.SU = SU
        # Load all systems
        sInds = SU.sInds
        if "nsystems" in params.keys():
            nsystems = params["nsystems"]
            sInds = sInds[:nsystems]

        self.systems = []
        for sInd in tqdm(sInds, desc="Loading systems", position=0, leave=False):
            # if cache:
            #     cache_file = Path(cache_base, system_file.stem + ".p")
            #     if cache_file.exists():
            #         with open(cache_file, "rb") as f:
            #             system = pickle.load(f)
            #     else:
            #         system = ExosimsSystem(system_file)
            #         with open(cache_file, "wb") as f:
            #             pickle.dump(system, f)
            system = ExosimsSystem(SU, sInd)
            self.systems.append(system)
            # else:
            #     system = ExosimsSystem(system_file)
            #     if system is not None:
            #         self.systems.append(system)

        # Get star ids
        self.ids = [system.star.id for system in self.systems]
        self.names = [system.star.name for system in self.systems]


class ExosimsSystem(System):
    """
    Class for the whole stellar system
    """

    def __init__(self, SU, sInd):

        # Create star object
        self.star = ExosimsStar(SU, sInd)
        self.planets = []
        self.pInds = np.where(SU.plan2star == sInd)[0]
        # loop over all planets
        for pInd in self.pInds:
            self.planets.append(ExosimsPlanet(SU, self.star, pInd))

        self.cleanup()
        # Set up rebound simulation
        # self.sim = rebound.Simulation()
        # self.sim.G = const.G.value
        # self.sim.add(
        #     m=self.star.mass.decompose().value,
        #     x=self.star._x[0].decompose().value,
        #     y=self.star._y[0].decompose().value,
        #     z=self.star._z[0].decompose().value,
        #     vx=self.star._vx[0].decompose().value,
        #     vy=self.star._vy[0].decompose().value,
        #     vz=self.star._vz[0].decompose().value,
        # )
        # for planet in self.planets:
        #     self.sim.add(
        #         m=planet.mass.decompose().value,
        #         x=planet._x[0].decompose().value,
        #         y=planet._y[0].decompose().value,
        #         z=planet._z[0].decompose().value,
        #         vx=planet._vx[0].decompose().value,
        #         vy=planet._vy[0].decompose().value,
        #         vz=planet._vz[0].decompose().value,
        #     )
        # self.sim.move_to_com()


class ExosimsStar(Star):
    """
    Class for the star in the EXOSIMS systems
    """

    def __init__(self, SU, sInd):

        # Get the object's data from the fits file

        TL = SU.TargetList
        # Time data
        self._t = 0 * u.yr

        # Position data
        self._x = 0 * u.AU
        self._y = 0 * u.AU
        self._z = 0 * u.AU

        # Velocity data
        self._vx = 0 * u.AU / u.yr
        self._vy = 0 * u.AU / u.yr
        self._vz = 0 * u.AU / u.yr

        # System identifiers
        self.id = sInd
        self.name = TL.Name[sInd]

        # System midplane information
        # self.midplane_PA = (obj_header["PA"] * u.deg).to(u.rad)  # Position angle
        # self.midplane_I = (obj_header["I"] * u.deg).to(u.rad)  # Inclination

        # Proper motion
        self.PMRA = TL.pmra[sInd] * u.mas / u.yr
        self.PMDEC = TL.pmdec[sInd] * u.mas / u.yr

        # Celestial coordinates
        self.coords = TL.coords[sInd]
        self.RA = self.coords.ra
        self.DEC = self.coords.dec
        self.dist = TL.dist[sInd]

        # Spectral properties
        self.spectral_type = TL.spectral_class[sInd]
        self.MV = TL.Vmag[sInd]  # Absolute V band mag

        # Commenting out for now, these are available but
        # not every star has all the information
        # self.Bmag = obj_header["BMAG"]
        # self.Vmag = obj_header["VMAG"]
        # self.Rmag = obj_header["RMAG"]
        # self.Imag = obj_header["IMAG"]
        # self.Jmag = obj_header["JMAG"]
        # self.Hmag = obj_header["HMAG"]
        # self.Kmag = obj_header["KMAG"]

        # Stellar properties
        self.Lstar = TL.L[sInd] * u.Lsun  # Bolometric luminosity
        # self.Teff = obj_header["TEFF"] * u.K  # Effective temperature
        # self.angdiam = obj_header["ANGDIAM"] * u.K  # Angular diameter
        self.mass = TL.MsTrue[sInd]
        # self.radius = obj_header["RSTAR"] * u.R_sun

        # Propagation table
        # self.vectors = pd.DataFrame(
        #     {
        #         "t": [self._t[0].decompose().value],
        #         "x": [self._x[0].decompose().value],
        #         "y": [self._y[0].decompose().value],
        #         "z": [self._z[0].decompose().value],
        #         "vx": [self._vx[0].decompose().value],
        #         "vy": [self._vy[0].decompose().value],
        #         "vz": [self._vz[0].decompose().value],
        #     }
        # )


class ExosimsPlanet(Planet):
    """
    Class for the planets in the EXOSIMS systems
    Assumes e=0, which is the case for all downloaded systems
    """

    def __init__(self, SU, star, pInd):

        # Time data
        self._t = 0 * u.yr
        # self.t0 = Time(self._t[0].to(u.yr).value, format="decimalyear")

        # Position data
        self._x = SU.r[pInd][0] * u.AU
        self._y = SU.r[pInd][1] * u.AU
        self._z = SU.r[pInd][2] * u.AU

        # Velocity data
        self._vx = SU.v[pInd][0] * u.AU / u.yr
        self._vy = SU.v[pInd][1] * u.AU / u.yr
        self._vz = SU.v[pInd][2] * u.AU / u.yr

        # Assign the planet's keplerian orbital elements
        self.a = SU.a[pInd]
        self.e = SU.e[pInd]
        self.i = SU.I[pInd]
        self.W = SU.O[pInd]
        # self.w = (obj_header["ARGPERI"] * u.deg).to(u.rad)
        self.w = SU.w[pInd]

        # Assign the planet's mass/radius information
        self.mass = SU.Mp[pInd]
        self.radius = SU.Rp[pInd]

        # Initial mean anomaly
        self.M0 = SU.M0[pInd]

        # Assign the planet's time-varying mean anomaly, argument of pericenter,
        # true anomaly, and contrast
        # self.rep_w = (obj_data[:, 7] * u.deg + 180 * u.deg) % (2 * np.pi * u.rad)
        # self.M = (obj_data[:, 8] * u.deg + 180 * u.deg) % (2 * np.pi * u.rad)
        # self.nu = (self.rep_w + self.M) % (
        #     2 * np.pi * u.rad
        # )  # true anomaly for circular orbits
        # self.contrast = obj_data[:, 15]

        # Gravitational parameter
        self.mu = (const.G * (self.mass + star.mass)).decompose()
        self.T = (2 * np.pi * np.sqrt(self.a**3 / self.mu)).to(u.d)
        self.w_p = self.w
        self.w_s = (self.w + np.pi * u.rad) % (2 * np.pi * u.rad)
        self.secosw = np.sqrt(self.e) * np.cos(self.w)
        self.sesinw = np.sqrt(self.e) * np.sin(self.w)

        # Because we have the mean anomaly at an epoch we can calculate the
        # time of periastron as t0 - T_e where T_e is the time since periastron
        # passage
        # T_e = (self.T * self.M0 / (2 * np.pi * u.rad)).decompose()
        # self.T_p = self.t0 - T_e

        # Calculate the time of conjunction
        # self.T_c = Time(
        #     rvo.timeperi_to_timetrans(
        #         self.T_p.jd, self.T.value, self.e, self.w_s.value
        #     ),
        #     format="jd",
        # )
        self.K = (
            (2 * np.pi * const.G / self.T) ** (1 / 3.0)
            * (self.mass * np.sin(self.i) / star.mass ** (2 / 3.0))
            * (1 - self.e**2) ** (-1 / 2)
        ).decompose()

        # Mean angular motion
        self.n = (np.sqrt(self.mu / self.a**3)).decompose()

        # Propagation table
        # self.vectors = pd.DataFrame(
        #     {
        #         "t": [self._t[0].decompose().value],
        #         "x": [self._x[0].decompose().value],
        #         "y": [self._y[0].decompose().value],
        #         "z": [self._z[0].decompose().value],
        #         "vx": [self._vx[0].decompose().value],
        #         "vy": [self._vy[0].decompose().value],
        #         "vz": [self._vz[0].decompose().value],
        #     }
        # )

        self.star = star
