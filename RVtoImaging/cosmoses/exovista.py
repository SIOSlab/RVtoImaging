import pickle
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io.fits import getdata, getheader
from astropy.time import Time
from tqdm import tqdm

from RVtoImaging.cosmos import Planet, Star, System, Universe
from RVtoImaging.utils import get_data


def create_universe(universe_params):
    data_path = Path(universe_params["data_path"])
    un = universe_params["universe_number"]
    full_path = f"{data_path}/{un}"
    if not Path(full_path).exists():
        get_data([un])
    universe = ExovistaUniverse(full_path, cache=True)
    return universe


class ExovistaUniverse(Universe):
    """
    Class for the whole exoVista universe
    """

    def __init__(self, path, cache=False):
        """
        Args:
            path (str or Path):
                Location of all the system files. Should be something like "data/1/"
        """
        self.type = "ExoVista"
        if cache:
            cache_base = Path(".cache", path.split("/")[1])
            if not cache_base.exists():
                cache_base.mkdir(parents=True)
        self.path = path

        # Load all systems
        p = Path(path).glob("*.fits")
        system_files = [x for x in p if x.is_file]
        self.systems = []
        for system_file in tqdm(
            system_files, desc="Loading systems", position=0, leave=False
        ):
            if cache:
                cache_file = Path(cache_base, system_file.stem + ".p")
                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        system = pickle.load(f)
                else:
                    system = ExovistaSystem(system_file)
                    with open(cache_file, "wb") as f:
                        pickle.dump(system, f)
                self.systems.append(system)
            else:
                system = ExovistaSystem(system_file)
                if system is not None:
                    self.systems.append(system)

        # Get star ids
        self.ids = [system.star.id for system in self.systems]
        self.names = [system.star.name for system in self.systems]

        Universe.__init__(self)


class ExovistaSystem(System):
    """
    Class for the whole stellar system
    """

    def __init__(self, infile):
        self.file = infile

        # fits file extensions, exoVista hard codes these
        planet_ext = 4

        # Get the number of planets
        with open(infile, "rb") as f:
            # read header of first extension
            h = getheader(f, ext=0, memmap=False)
        n_ext = h["N_EXT"]  # get the largest extension
        nplanets = n_ext - 3

        # Create star object
        self.star = ExovistaStar(infile)
        self.planets = []
        # loop over all planets
        for i in range(nplanets):
            self.planets.append(ExovistaPlanet(infile, planet_ext + i, self.star))

        # self.cleanup()

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


class ExovistaPlanet(Planet):
    """
    Class for the planets in the exoVista systems
    Assumes e=0, which is the case for all downloaded systems
    """

    def __init__(self, infile, fits_ext, star):

        # Get the object's data from the fits file
        with open(infile, "rb") as f:
            obj_data, obj_header = getdata(f, ext=fits_ext, header=True, memmap=False)

        self.star = star
        # Time data, setting default epoch to the year 2000
        self._t = obj_data[:, 0] * u.yr
        self.t0 = Time(self._t[0].to(u.yr).value + 2000, format="decimalyear")

        # Position data
        self._x = obj_data[:, 9] * u.AU
        self._y = obj_data[:, 10] * u.AU
        self._z = obj_data[:, 11] * u.AU

        # Velocity data
        self._vx = obj_data[:, 12] * u.AU / u.yr
        self._vy = obj_data[:, 13] * u.AU / u.yr
        self._vz = obj_data[:, 14] * u.AU / u.yr
        # Assign the planet's time-varying mean anomaly, argument of pericenter,
        # true anomaly, and contrast
        self.rep_w = (obj_data[:, 7] * u.deg + 180 * u.deg) % (2 * np.pi * u.rad)
        self.M = (obj_data[:, 8] * u.deg + 180 * u.deg) % (2 * np.pi * u.rad)
        self.nu = (self.rep_w + self.M) % (
            2 * np.pi * u.rad
        )  # true anomaly for circular orbits
        self.contrast = obj_data[:, 15]

        # Initial mean anomaly
        self.M0 = self.nu[0]
        planet_dict = {
            "t0": self.t0,
            "a": obj_header["A"] * u.AU,
            "e": obj_header["E"],
            "inc": (obj_header["I"] * u.deg).to(u.rad) + star.midplane_I,
            "W": (obj_header["LONGNODE"] * u.deg).to(u.rad),
            "w": 0 * u.rad,
            "mass": obj_header["M"] * u.M_earth,
            "radius": obj_header["R"] * u.R_earth,
            "M0": self.M0,
            "p": 0.2,
        }
        Planet.__init__(self, planet_dict)
        self.solve_dependent_params()

        # Assign the planet's keplerian orbital elements
        # self.a = obj_header["A"] * u.AU
        # self.e = obj_header["E"]
        # self.inc = (obj_header["I"] * u.deg).to(u.rad)
        # self.W = (obj_header["LONGNODE"] * u.deg).to(u.rad)
        # # self.w = (obj_header["ARGPERI"] * u.deg).to(u.rad)
        # self.w = 0 * u.rad

        # # Assign the planet's mass/radius information
        # self.mass = obj_header["M"] * u.M_earth
        # self.radius = obj_header["R"] * u.R_earth

        # # Gravitational parameter
        # self.mu = (const.G * (self.mass + star.mass)).decompose()
        # self.T = (2 * np.pi * np.sqrt(self.a**3 / self.mu)).to(u.d)
        # self.w_p = self.w
        # self.w_s = (self.w + np.pi * u.rad) % (2 * np.pi * u.rad)
        # self.secosw = np.sqrt(self.e) * np.cos(self.w)
        # self.sesinw = np.sqrt(self.e) * np.sin(self.w)

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
        # self.K = (
        #     (2 * np.pi * const.G / self.T) ** (1 / 3.0)
        #     * (self.mass * np.sin(self.inc) / star.mass ** (2 / 3.0))
        #     * (1 - self.e**2) ** (-1 / 2)
        # ).decompose()

        # # Mean angular motion
        # self.n = (np.sqrt(self.mu / self.a**3)).decompose() * u.rad

        # Propagation table
        self.vectors = pd.DataFrame(
            {
                "t": [self._t[0].decompose().value],
                "x": [self._x[0].decompose().value],
                "y": [self._y[0].decompose().value],
                "z": [self._z[0].decompose().value],
                "vx": [self._vx[0].decompose().value],
                "vy": [self._vy[0].decompose().value],
                "vz": [self._vz[0].decompose().value],
            }
        )

        self.star = star

        # self.classify_planet()


class ExovistaStar(Star):
    """
    Class for the star in the exoVista systems
    """

    def __init__(self, infile):

        # Get the object's data from the fits file
        with open(infile, "rb") as f:
            obj_data, obj_header = getdata(f, ext=3, header=True, memmap=False)

        # Time data
        self._t = obj_data[:, 0] * u.yr

        # Position data
        self._x = obj_data[:, 9] * u.AU
        self._y = obj_data[:, 10] * u.AU
        self._z = obj_data[:, 11] * u.AU

        # Velocity data
        self._vx = obj_data[:, 12] * u.AU / u.yr
        self._vy = obj_data[:, 13] * u.AU / u.yr
        self._vz = obj_data[:, 14] * u.AU / u.yr

        # System identifiers
        self.id = obj_header["STARID"]
        self.name = f"HIP {obj_header['HIP']}"

        # System midplane information
        self.midplane_PA = (obj_header["PA"] * u.deg).to(u.rad)  # Position angle
        self.midplane_I = np.abs((obj_header["I"] * u.deg).to(u.rad))  # Inclination
        # if self.midplane_I < 0:
        #     breakpoint()

        # Proper motion
        self.PMRA = obj_header["PMRA"] * u.mas / u.yr
        self.PMDEC = obj_header["PMDEC"] * u.mas / u.yr

        # Celestial coordinates
        self.RA = obj_header["RA"] * u.deg
        self.DEC = obj_header["DEC"] * u.deg
        self.dist = obj_header["DIST"] * u.pc
        self.coords = SkyCoord(ra=self.RA, dec=self.DEC, distance=self.dist)

        # Spectral properties
        self.spectral_type = obj_header["TYPE"]
        self.MV = obj_header["M_V"]  # Absolute V band mag

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
        self.Lstar = obj_header["LSTAR"] * u.Lsun  # Bolometric luminosity
        self.Teff = obj_header["TEFF"] * u.K  # Effective temperature
        self.angdiam = obj_header["ANGDIAM"]  # Angular diameter
        self.mass = obj_header["MASS"] * u.M_sun
        self.radius = obj_header["RSTAR"] * u.R_sun
        self.mu = self.mass * const.G

        # Propagation table
        self.vectors = pd.DataFrame(
            {
                "t": [self._t[0].decompose().value],
                "x": [self._x[0].decompose().value],
                "y": [self._y[0].decompose().value],
                "z": [self._z[0].decompose().value],
                "vx": [self._vx[0].decompose().value],
                "vy": [self._vy[0].decompose().value],
                "vz": [self._vz[0].decompose().value],
            }
        )
