import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from keplertools import fun as kt
from tqdm import tqdm


class Universe:
    """
    The base class for universe. Keeps track of the planetary systems.
    """

    def __init__(self) -> None:
        pass


class System:
    """
    Class for a single system. Must have a star and a list of planets.
    """

    def __init__(self) -> None:
        pass

    def cleanup(self):
        # Sort the planets in the system by semi-major axis
        a_vals = [planet.a.value for planet in self.planets]
        self.planets = np.array(self.planets)[np.argsort(a_vals)].tolist()
        self.pInds = self.pInds[np.argsort(a_vals)]

    @property
    def a(self):
        return [planet.a for planet in self.planets]

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


class Planet:
    """
    Class for a planet
    """

    def __init__(self) -> None:
        pass

    def calc_vectors(self, t, return_r=True, return_v=False):
        """
        Given a time, calculate the planet's position and/or velocity vectors
        Args:
            t (Time):
                Time to calculate the position vectors at
        Returns:
            r(ndarray):
                3 x n stacked position vector
            v(ndarray):
                3 x n stacked velocity vector

        """
        # This will find the radial and velocity vectors at an epoch
        M = self.mean_anom(t)
        E = kt.eccanom(M.value, self.e)
        a, e, Omega, inc, w = self.a.decompose(), self.e, self.W, self.i, self.w
        if not np.isscalar(E):
            a = np.ones(len(E)) * a
            e = np.ones(len(E)) * e
            Omega = np.ones(len(E)) * Omega
            inc = np.ones(len(E)) * inc
            w = np.ones(len(E)) * w
        A = np.vstack(
            (
                a
                * (np.cos(Omega) * np.cos(w) - np.sin(Omega) * np.cos(inc) * np.sin(w)),
                a
                * (np.sin(Omega) * np.cos(w) + np.cos(Omega) * np.cos(inc) * np.sin(w)),
                a * np.sin(inc) * np.sin(w),
            )
        )

        B = np.vstack(
            (
                -a
                * np.sqrt(1 - e**2)
                * (np.cos(Omega) * np.sin(w) + np.sin(Omega) * np.cos(inc) * np.cos(w)),
                a
                * np.sqrt(1 - e**2)
                * (
                    -np.sin(Omega) * np.sin(w) + np.cos(Omega) * np.cos(inc) * np.cos(w)
                ),
                a * np.sqrt(1 - e**2) * np.sin(inc) * np.cos(w),
            )
        )

        # Calculate position vectors
        if return_r:
            if np.isscalar(self.mu) and not (np.isscalar(E)):
                r = np.matmul(A, np.array((np.cos(E) - e), ndmin=2)) + np.matmul(
                    B, np.array(np.sin(E), ndmin=2)
                )
            else:
                r = np.matmul(A, np.diag(np.cos(E) - e)) + np.matmul(
                    B, np.diag(np.sin(E))
                )

        # Calculate velocity vectors
        if return_v:
            if np.isscalar(self.mu) and not (np.isscalar(E)):
                v = (
                    np.matmul(-A, np.array(np.sin(E), ndmin=2))
                    + np.matmul(B, np.array(np.cos(E), ndmin=2))
                ) * np.tile(
                    np.sqrt(self.mu * a ** (-3.0)) / (1 - e * np.cos(E)), (3, 1)
                )
            else:
                v = np.matmul(
                    np.matmul(-A, np.diag(np.sin(E)))
                    + np.matmul(B, np.diag(np.cos(E))),
                    np.diag(np.sqrt(self.mu * a ** (-3.0)) / (1 - e * np.cos(E))),
                )

        # Rotate around x axis with midplane inclination
        rot1 = np.array(
            [
                [1, 0, 0],
                [
                    0,
                    np.cos(self.star.midplane_I.to(u.rad)),
                    -np.sin(self.star.midplane_I.to(u.rad)),
                ],
                [
                    0,
                    np.sin(self.star.midplane_I.to(u.rad)),
                    np.cos(self.star.midplane_I.to(u.rad)),
                ],
            ]
        )

        # Rotate around z axis with midplane position angle
        rot2 = np.array(
            [
                [
                    np.cos(self.star.midplane_PA.to(u.rad)),
                    np.sin(self.star.midplane_PA.to(u.rad)),
                    0,
                ],
                [
                    -np.sin(self.star.midplane_PA.to(u.rad)),
                    np.cos(self.star.midplane_PA.to(u.rad)),
                    0,
                ],
                [0, 0, 1],
            ]
        )
        if return_r:
            r = np.matmul(r.T, rot1)
            r = np.matmul(r, rot2)
        if return_v:
            v = np.matmul(v.T, rot1)
            v = np.matmul(v, rot2)

        if return_r and return_v:
            return r, v
        if return_r:
            return r
        if return_v:
            return v

    def calc_vs(self, t, return_nu=False):
        """
        Calculate the radial velocities

        Args:
            t (astropy Time):
                input time
            return_nu (Bool):
                Whether the true anomaly should be returned

        Returns:
            vs (Astropy quantity):
                The radial velocity at time t
            nu (astropy Quantity):
                True anomaly
        """
        M = self.mean_anom(t)
        E = kt.eccanom(M, self.e)
        nu = kt.trueanom(E, self.e) * u.rad
        vs = (
            np.sqrt(
                const.G / ((self.mass + self.star.mass) * self.a * (1 - self.e**2))
            )
            * self.mass
            * np.sin(self.i)
            * (np.cos(self.w + nu) + self.e * np.cos(self.w))
        )
        if return_nu:
            return vs.decompose(), nu
        else:
            return vs.decompose()

    def mean_anom(self, t):
        """
        Calculate the mean anomaly at a given time, assumes initial time is 0
        Args:
            t (Time):
                Time to calculate mean anomaly

        Returns:
            M (astropy Quantity):
                Planet's mean anomaly at t (radians)
        """
        # t is a Time quantity
        # n is in standard SI units so the times are converted to seconds
        M1 = (self.n * t).decompose() * u.rad
        M = ((M1 + self.M0).to(u.rad)) % (2 * np.pi * u.rad)
        return M

    def classify_planet(self):
        """
        This determines the Kopparapu bin of the planet This is adapted from
        the EXOSIMS SubtypeCompleteness method classifyPlanets so that EXOSIMS
        isn't a mandatory import
        """
        # Calculate the luminosity of the star, assuming main-sequence
        if self.mass < 2 * u.M_sun:
            self.Ls = const.L_sun * (self.star.mass / const.M_sun) ** 4
        else:
            self.Ls = 1.4 * const.L_sun * (self.star.mass / const.M_sun) ** 3.5

        Rp = self.radius.to("earthRad").value
        # a = self.a.to("AU").value
        # e = self.e

        # Find the stellar flux at the planet's location as a fraction of earth's
        earth_Lp = const.L_sun / (1 * (1 + (0.0167**2) / 2)) ** 2
        self.Lp = (
            self.Ls / (self.a.to("AU").value * (1 + (self.e**2) / 2)) ** 2 / earth_Lp
        )

        # Find Planet Rp range
        Rp_bins = np.array([0, 0.5, 1.0, 1.75, 3.5, 6.0, 14.3, 11.2 * 4.6])
        # Rp_lo = Rp_bins[:-1]
        # Rp_hi = Rp_bins[1:]
        Rp_types = [
            "Sub-Rocky",
            "Rocky",
            "Super-Earth",
            "Sub-Neptune",
            "Sub-Jovian",
            "Jovian",
            "Super-Jovian",
        ]
        self.L_bins = np.array(
            [
                [1000, 182, 1.0, 0.28, 0.0035, 5e-5],
                [1000, 182, 1.0, 0.28, 0.0035, 5e-5],
                [1000, 187, 1.12, 0.30, 0.0030, 5e-5],
                [1000, 188, 1.15, 0.32, 0.0030, 5e-5],
                [1000, 220, 1.65, 0.45, 0.0030, 5e-5],
                [1000, 220, 1.65, 0.40, 0.0025, 5e-5],
                [1000, 220, 1.68, 0.45, 0.0025, 5e-5],
                [1000, 220, 1.68, 0.45, 0.0025, 5e-5],
            ]
        )
        # self.L_bins = np.array(
        #     [
        #         [1000, 182, 1.0, 0.28, 0.0035, 5e-5],
        #         [1000, 187, 1.12, 0.30, 0.0030, 5e-5],
        #         [1000, 188, 1.15, 0.32, 0.0030, 5e-5],
        #         [1000, 220, 1.65, 0.45, 0.0030, 5e-5],
        #         [1000, 220, 1.65, 0.40, 0.0025, 5e-5],
        #     ]
        # )

        # Find the bin of the radius
        self.Rp_bin = np.digitize(Rp, Rp_bins) - 1
        try:
            self.Rp_type = Rp_types[self.Rp_bin]
        except IndexError:
            print(f"Error handling Rp_type of planet with Rp_bin of {self.Rp_bin}")
            self.Rp_type = None

        # TODO Fix this to give correct when at edge cases since technically
        # they're not straight lines

        # index of planet temp. cold,warm,hot
        L_types = ["Very Hot", "Hot", "Warm", "Cold", "Very Cold"]
        specific_L_bins = self.L_bins[self.Rp_bin, :]
        self.L_bin = np.digitize(self.Lp.decompose().value, specific_L_bins) - 1
        try:
            self.L_type = L_types[self.L_bin]
        except IndexError:
            print(f"Error handling L_type of planet with L_bin of {self.L_bin}")


class Star:
    """
    The star of a system
    """

    def __init__(self):
        pass
