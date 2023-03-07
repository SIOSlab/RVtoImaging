import copy

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
import rebound
from astropy.time import Time
from keplertools import fun as kt
from tqdm import tqdm

import radvel.orbit as rvo
import radvel.utils as rvu


class Universe:
    """
    The base class for universe. Keeps track of the planetary systems.
    """

    def __init__(self) -> None:
        pass

    def __repr__(self):
        str = f"{self.type} universe\n"
        str += f"{len(self.systems)} systems loaded"
        return str


class System:
    """
    Class for a single system. Must have a star and a list of planets.
    """

    def __init__(self) -> None:
        pass

    def __repr__(self):
        return (
            f"{self.star.name}\tdist:{self.star.dist}\t"
            f"Type:{self.star.spectral_type}\n\n"
            f"Planets:\n{self.get_p_df()}"
        )

    def cleanup(self):
        # Sort the planets in the system by semi-major axis
        a_vals = [planet.a.value for planet in self.planets]
        self.planets = np.array(self.planets)[np.argsort(a_vals)].tolist()
        self.pInds = self.pInds[np.argsort(a_vals)]

    def getpattr(self, attr):
        # Return array of all planet's attribute value, e.g. all semi-major
        # axis values
        if type(getattr(self.planets[0], attr)) == u.Quantity:
            return [getattr(planet, attr).value for planet in self.planets] * getattr(
                self.planets[0], attr
            ).unit
        else:
            return [getattr(planet, attr) for planet in self.planets]

    def get_p_df(self):
        patts = [
            "K",
            "T",
            "secosw",
            "sesinw",
            "T_c",
            "a",
            "e",
            "inc",
            "W",
            "w",
            "M0",
            "t0",
            "mass",
            "radius",
        ]
        p_df = pd.DataFrame()
        for att in patts:
            pattr = self.getpattr(att)
            if type(pattr) == u.Quantity:
                p_df[att] = pattr.value
            else:
                p_df[att] = pattr

        return p_df

    def propagate(self, times):
        """
        Propagates system at all times given. Currently does not handle
        """
        # Get unique time values, multiple instruments can be scheduled to
        # observe at the same time
        times = Time(np.unique(times.jd), format="jd")
        # self.rv_vals = np.zeros(len(times)) * u.m / u.s
        # for planet in self.planets:
        #     M = planet.mean_anom(times)
        #     E = kt.eccanom(M, planet.e)
        #     nu = kt.trueanom(E, planet.e) * u.rad
        #     planet.rv_times = times
        #     planet.nu = nu
        #     planet.rv_vals = -planet.K * (
        #         np.cos(planet.w + nu) + planet.e * np.cos(planet.w)
        #     )
        #     self.rv_vals += planet.rv_vals

        syst_M = []
        for planet in self.planets:
            syst_M.append(planet.mean_anom(times))
        Marr = np.stack(syst_M).value
        self.rv_vals = (
            -kt.calc_RV_from_M(
                Marr,
                self.getpattr("e"),
                self.getpattr("w").to(u.rad).value,
                self.getpattr("K").value,
            )
            * u.m
            / u.s
        )
        # E = kt.eccanom(M, planet.e)
        # nu = kt.trueanom(E, planet.e) * u.rad
        # planet.rv_times = times
        # planet.nu = nu
        # planet.rv_vals = -planet.K * (
        #     np.cos(planet.w + nu) + planet.e * np.cos(planet.w)
        # )
        # self.rv_vals += planet.rv_vals

        # Storing as dataframe too
        rv_df = pd.DataFrame(
            np.stack((times, self.rv_vals.value), axis=-1), columns=["t", "rv"]
        )
        # Keep track in case future propagation is necessary
        # if hasattr(self, "rv_df"):
        #     # Add the new rv values to rv_df and then sort
        #     self.rv_df = (
        #         pd.concat([self.rv_df, rv_df]).drop_duplicates().sort_values(by="t")
        #     )
        # else:
        self.rv_df = rv_df

        # Rebound stuff, not needed right now
        # self.instantiate_rebound()
        # rebound_times = ((times.jd - planet.t0.jd) * u.d).to(u.s).value
        # self.propagate_system(rebound_times)

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

    def instantiate_rebound(self):
        # Set up rebound simulation
        self.sim = rebound.Simulation()
        self.sim.G = const.G.value
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
        self.sim.add(
            m=self.star.mass.decompose().value,
            x=self.star._x.decompose().value,
            y=self.star._y.decompose().value,
            z=self.star._z.decompose().value,
            vx=self.star._vx.decompose().value,
            vy=self.star._vy.decompose().value,
            vz=self.star._vz.decompose().value,
        )
        for planet in self.planets:
            self.sim.add(
                m=planet.mass.decompose().value,
                x=planet._x.decompose().value,
                y=planet._y.decompose().value,
                z=planet._z.decompose().value,
                vx=planet._vx.decompose().value,
                vy=planet._vy.decompose().value,
                vz=planet._vz.decompose().value,
            )
        self.sim.move_to_com()

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

    def __init__(self, planet_dict) -> None:
        for att, value in planet_dict.items():
            setattr(self, att, value)
        # self.solve_dependent_params()

    def dump_params(self):
        params = {
            "t0": self.t0,
            "a": self.a,
            "e": self.e,
            "mass": self.mass,
            "radius": self.radius,
            "inc": self.inc,
            "W": self.W,
            "w": self.w,
            "M0": self.M0,
            "p": self.p,
        }
        return params

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
        E = kt.eccanom(M.to(u.rad).value, self.e)
        a, e, Omega, inc, w = (
            self.a.decompose(),
            self.e,
            self.W.to(u.rad).value,
            self.inc.to(u.rad).value,
            self.w.to(u.rad).value,
        )
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
            return r

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

    def mean_anom(self, times):
        """
        Calculate the mean anomaly at the given times
        Args:
            times (astropy Time array):
                Times to calculate mean anomaly

        Returns:
            M (astropy Quantity array):
                Planet's mean anomaly at t (radians)
        """
        M = ((self.n * ((times.jd - self.t0.jd) * u.d)).decompose() + self.M0) % (
            2 * np.pi * u.rad
        )
        return M

    def solve_dependent_params(self):
        self.mu = (const.G * (self.mass + self.star.mass)).decompose()
        self.T = (2 * np.pi * np.sqrt(self.a**3 / self.mu)).to(u.d)
        self.w_p = self.w
        self.w_s = (self.w + np.pi * u.rad) % (2 * np.pi * u.rad)
        self.secosw = np.sqrt(self.e) * np.cos(self.w)
        self.sesinw = np.sqrt(self.e) * np.sin(self.w)
        T_e = (self.T * self.M0 / (2 * np.pi * u.rad)).decompose()
        self.T_p = self.t0 - T_e

        # Calculate the time of conjunction
        self.T_c = Time(
            rvo.timeperi_to_timetrans(
                self.T_p.jd, self.T.value, self.e, self.w_s.value
            ),
            format="jd",
        )
        self.K = (
            (2 * np.pi * const.G / self.T) ** (1 / 3.0)
            * (self.mass * np.sin(self.inc) / self.star.mass ** (2 / 3.0))
            * (1 - self.e**2) ** (-1 / 2)
        ).decompose()

        # Mean angular motion
        self.n = (np.sqrt(self.mu / self.a**3)).decompose() * u.rad

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


class FitSystem(System):
    def __init__(self, search_obj, true_system):
        """
        This function creates a rvtools system object with the planets found
        during the fitting process. It also calculates the planet that the
        fit is most likely describing.
        """
        search_params = search_obj.post.params
        self.star = copy.deepcopy(true_system.star)
        self.planets = []
        for nplan in range(1, search_obj.num_planets + 1):
            # Create dictionary to initialize planet with
            plan_dict = {}
            plan_dict["T"] = search_params[f"per{nplan}"].value * u.d
            plan_dict["T_c"] = Time(search_params[f"tc{nplan}"].value, format="jd")
            plan_dict["secosw"] = search_params[f"secosw{nplan}"].value
            plan_dict["sesinw"] = search_params[f"sesinw{nplan}"].value
            plan_dict["K"] = search_params[f"k{nplan}"].value * u.m / u.s

            # Initialize the planet object
            planet = FitPlanet(plan_dict, true_system)
            self.planets.append(planet)
        self.true_system = true_system

    def get_p_df(self):
        patts = [
            "K",
            "T",
            "secosw",
            "sesinw",
            "T_c",
            "a",
            "e",
            "w",
            "M0",
            "t0",
            "msini",
            "best_match",
            "best_rms",
        ]
        p_df = pd.DataFrame()
        for att in patts:
            pattr = self.getpattr(att)
            if type(pattr) == u.Quantity:
                p_df[att] = pattr.value
            else:
                p_df[att] = pattr

        return p_df


class FitPlanet(Planet):
    def __init__(self, planet_dict, true_system):
        """
        filling in parameters that are not found in the fitting process.
        Assumes that the planet_dict includes
        T, Tc, secosw, sesinw, K
        basis.
        """
        self.star = true_system.star
        for att, value in planet_dict.items():
            setattr(self, att, value)
        self.e = self.secosw**2 + self.sesinw**2
        self.w_s = np.arctan2(self.sesinw, self.secosw) * u.rad
        self.w = (self.w_s + np.pi * u.rad) % (2 * np.pi * u.rad)
        self.w_p = self.w
        self.T_p = rvo.timetrans_to_timeperi(
            self.T_c.jd, self.T.to(u.d).value, self.e, self.w_s.to(u.rad).value
        )
        self.msini = (
            rvu.Msini(
                self.K.decompose().value,
                self.T.to(u.d).value,
                self.star.mass.to(u.M_sun).value,
                self.e,
            )
            * u.M_earth
        )
        self.mu = (const.G * self.star.mass).decompose()
        self.a = ((self.mu * (self.T / (2 * np.pi)) ** 2) ** (1 / 3)).decompose()

        # Finding the mean anomaly at time of conjunction
        nu_p = (np.pi / 2 * u.rad - self.w_s) % (2 * np.pi * u.rad)
        E_p = 2 * np.arctan2(
            np.sqrt((1 - self.e)) * np.tan(nu_p / 2), np.sqrt((1 + self.e))
        )
        self.M0 = (E_p - self.e * np.sin(E_p) * u.rad) % (2 * np.pi * u.rad)
        self.t0 = Time(self.T_c.jd, format="jd")

        self.compare_to_system(true_system)

    def compare_to_system(self, true_system):
        """
        Function to check how well a planet fit compares to the true system.
        Gets the index of the planet that the fit is most likely describing
        and the RMS error of the fit.
        """
        # Not doing argument of periastron because it is finicky for circular
        # orbits
        # werr = true_system.getpattr('w') - fitted_planet.w
        eerr = true_system.getpattr("e") - self.e
        Kerr = true_system.getpattr("K") - self.K
        Terr = true_system.getpattr("T") - self.T

        # Normalize the differences by the range of the system values
        Knorm = Kerr / (np.ptp(true_system.getpattr("K")))
        Tnorm = Terr / (np.ptp(true_system.getpattr("T")))

        self.all_rms = np.sqrt(eerr**2 + Knorm**2 + Tnorm**2)
        self.best_match = np.argmin(self.all_rms)
        self.best_rms = self.all_rms[self.best_match]
