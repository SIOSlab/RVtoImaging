import json
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

from EXOSIMS.util.get_module import get_module_from_specs
from RVtools.cosmos import Planet, Star, System, Universe


def create_universe(universe_params):
    script_path = Path(universe_params["script"])
    with open(script_path) as f:
        specs = json.loads(f.read())
    assert "seed" in specs.keys(), (
        "For reproducibility the seed should" " not be randomized by EXOSIMS."
    )

    # Need to use SurveySimulation if we want to have a random seed
    SS = get_module_from_specs(specs, "SurveySimulation")(**specs)
    SU = SS.SimulatedUniverse
    universe_params["missionStart"] = specs["missionStart"]
    # SU = get_module_from_specs(specs, "SimulatedUniverse")(**specs)
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
        self.type = "EXOSIMS"
        self.SU = SU
        self.t0 = Time(params["missionStart"], format="mjd")
        # Load all systems
        sInds = SU.sInds
        if "nsystems" in params.keys():
            nsystems = params["nsystems"]
            sInds = sInds[:nsystems]
        else:
            sInds = SU.sInds
            nsystems = len(sInds)

        if "cache_path" in params.keys():
            self.cache_path = Path(params["cache_path"])

        self.systems = []
        for sInd in tqdm(sInds, desc="Loading systems", position=0, leave=False):
            system = ExosimsSystem(SU, sInd, self.t0)
            self.systems.append(system)

        # Get star ids
        self.ids = [system.star.id for system in self.systems]
        self.names = [system.star.name for system in self.systems]

        Universe.__init__(self)


class ExosimsSystem(System):
    """
    Class for the whole stellar system
    """

    def __init__(self, SU, sInd, t0):

        # Create star object
        self.star = ExosimsStar(SU, sInd)
        self.planets = []
        self.pInds = np.where(SU.plan2star == sInd)[0]
        # loop over all planets
        for pInd in self.pInds:
            self.planets.append(ExosimsPlanet(SU, self.star, pInd, t0))

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
        self.name = TL.Name[sInd].replace(" ", "_")

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
        self.mu = self.mass * const.G
        # self.radius = obj_header["RSTAR"] * u.R_sun

        # Propagation table
        self.vectors = pd.DataFrame(
            {
                "t": [self._t.decompose().value],
                "x": [self._x.decompose().value],
                "y": [self._y.decompose().value],
                "z": [self._z.decompose().value],
                "vx": [self._vx.decompose().value],
                "vy": [self._vy.decompose().value],
                "vz": [self._vz.decompose().value],
            }
        )


class ExosimsPlanet(Planet):
    """
    Class for the planets in the EXOSIMS systems
    Assumes e=0, which is the case for all downloaded systems
    """

    def __init__(self, SU, star, pInd, t0):
        self.star = star

        # Time data
        self._t = t0.decimalyear * u.yr
        self.t0 = t0

        # Position data
        self._x = SU.r[pInd][0]
        self._y = SU.r[pInd][1]
        self._z = SU.r[pInd][2]

        # Velocity data
        self._vx = SU.v[pInd][0]
        self._vy = SU.v[pInd][1]
        self._vz = SU.v[pInd][2]

        # Assign the planet's keplerian orbital elements
        # self.a = SU.a[pInd]
        # self.e = SU.e[pInd]
        # self.inc = SU.I[pInd]
        # self.W = SU.O[pInd]
        # self.w = SU.w[pInd]

        # # Assign the planet's mass/radius information
        # self.mass = SU.Mp[pInd]
        # self.radius = SU.Rp[pInd]
        # # Initial mean anomaly
        # self.M0 = SU.M0[pInd]

        planet_dict = {
            "t0": t0,
            "a": SU.a[pInd],
            "e": SU.e[pInd],
            "inc": SU.I[pInd],
            "W": SU.O[pInd],
            "w": SU.w[pInd],
            "mass": SU.Mp[pInd],
            "radius": SU.Rp[pInd],
            "M0": SU.M0[pInd],
            "p": 0.2,
        }
        Planet.__init__(self, planet_dict)
        self.solve_dependent_params()
        # Propagation table
        self.vectors = pd.DataFrame(
            {
                "t": [self._t.decompose().value],
                "x": [self._x.decompose().value],
                "y": [self._y.decompose().value],
                "z": [self._z.decompose().value],
                "vx": [self._vx.decompose().value],
                "vy": [self._vy.decompose().value],
                "vz": [self._vz.decompose().value],
            }
        )
