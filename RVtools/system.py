import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io.fits import getheader
from astropy.time import Time
from tqdm import tqdm

from RVtools.planet import Planet
from RVtools.star import Star


class System:
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
        self.star = Star(infile)
        self.planets = []
        # loop over all planets
        for i in range(nplanets):
            self.planets.append(Planet(infile, planet_ext + i, self.star))

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

    def propagate_system(self, t):
        """
        Propage system with rebound
        """
        for time in tqdm(t, desc="System propagation"):
            if time in self.star.vectors["t"].values:
                continue
            self.sim.integrate(time)
            for j, p in enumerate(self.sim.particles):
                p_vectors = {
                    "t": [time],
                    "x": [p.x],
                    "y": [p.y],
                    "z": [p.z],
                    "vx": [p.vx],
                    "vy": [p.vy],
                    "vz": [p.vz],
                }
                if j == 0:
                    self.star.vectors = pd.concat(
                        [self.star.vectors, pd.DataFrame(p_vectors)]
                    )
                else:
                    planet = self.planets[j - 1]
                    planet.vectors = pd.concat(
                        [planet.vectors, pd.DataFrame(p_vectors)]
                    )
        self.star.vectors.sort_values("t", inplace=True)
        self.star.vectors.reset_index(drop=True, inplace=True)
        for planet in self.planets:
            planet.vectors.sort_values("t", inplace=True)
            planet.vectors.reset_index(drop=True, inplace=True)

    def simulate_rv_observations(self, times, error):

        # Propagate system so the star has the true RV values
        self.propagate_system(times)

        rv_error = error.decompose().value
        # Save the rv observation times
        self.rv_observation_times = Time(times, format="jd")
        # Create a dataframe to hold the planet data
        column_names = ["time", "truevel", "tel", "svalue", "time_year"]

        rv_data_df = pd.DataFrame(
            0, index=np.arange(len(times)), columns=column_names, dtype=object
        )
        # nu_array = np.zeros([len(times), 2])
        nu_array = np.zeros(len(times))

        star_df = self.star.vectors[self.star.vectors["t"].isin(times)]

        # Loop through the times and calculate the radial velocity at the desired time
        for i, row in star_df.iterrows():
            # M = planet.mean_anom(t)
            # E = kt.eccanom(M.value, planet.e)
            # nu = kt.trueanom(E, planet.e) * u.rad
            t_yr = (row.t * u.s).to(u.yr).value
            rv_data_df.at[i, "time"] = Time(
                t_yr, format="decimalyear"
            ).jd  # Time of observation in julian days
            rv_data_df.at[i, "time_year"] = t_yr
            # Velocity at observation in m/s
            rv_data_df.at[i, "truevel"] = row.vz
            # This is saying it's all the same inst
            rv_data_df.at[i, "tel"] = "i"

            # appending to nu array
            # nu_array[i] = nu[0].to(u.rad).value
        # Calculate a random velocity offset or error based on the rv uncertainty
        vel_offset = np.random.normal(scale=rv_error, size=len(times))
        vel_offset_df = pd.DataFrame({"err_offset": vel_offset})
        rv_data_df = pd.concat([rv_data_df, vel_offset_df], axis=1)  # Append

        # This is simply an array of the one sigma error with some noise added
        errvel = np.ones(len(times)) * rv_error + np.random.normal(
            scale=rv_error / 10, size=len(times)
        )
        errvel_df = pd.DataFrame({"errvel": errvel})
        rv_data_df = pd.concat([rv_data_df, errvel_df], axis=1)

        # Add the errors onto the velocities
        adjusted_vels = rv_data_df["truevel"] + vel_offset
        vel_df = pd.DataFrame({"mnvel": adjusted_vels})
        rv_data_df = pd.concat([rv_data_df, vel_df], axis=1)

        # Add true anomaly
        nu_df = pd.DataFrame({"nu": nu_array})
        rv_data_df = pd.concat([rv_data_df, nu_df], axis=1)

        # Force floats for float columns
        columns = ["time", "mnvel", "truevel", "errvel", "nu", "time_year"]
        rv_data_df[columns] = rv_data_df[columns].apply(pd.to_numeric)
        return rv_data_df
