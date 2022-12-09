import pickle
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from keplertools import fun as kt
from tqdm import tqdm

import radvel.orbit as rvo
import radvel.utils as rvu
import RVtools.utils as utils
from RVtools.logger import logger


class PDet:
    """
    Base class to do probability of detection calculations
    """

    def __init__(self, params, orbitfit, universe, library):
        self.method = params["construction_method"]
        self.systems_of_interest = params["systems_of_interest"]
        self.n_fits = params["number_of_orbits"]
        start_time = params["start_time"]
        end_time = params["end_time"]
        self.pdet_times = Time(np.arange(start_time.jd, end_time.jd, 25), format="jd")
        self.cov_samples = params["cov_samples"]

        self.pops = {}

        # Loop through all the systems we want to calculate probability of
        # detection for
        # if system_id in orbitfit.paths.keys():
        # system_path = orbitfit.paths[system_id]
        # planets_fitted = orbitfit.planets_fitted[system_id]
        # Check for the chains
        for system_id in self.systems_of_interest:
            system_path = (
                f"{orbitfit.cache_dir}/{universe.systems[system_id].star.name}"
            )
            _, prev_max, _ = library.check_orbitfit_dir(system_path)
            fit_path = Path(system_path, f"{prev_max}_depth")
            chains_path = Path(fit_path, "chains.csv.tar.bz2")
            search_path = Path(fit_path, "search.pkl")
            breakpoint()
            if chains_path.exists():
                logger.info(
                    (
                        "Loading chains and search data for "
                        f"{universe.names[system_id]}."
                    )
                )
                chains = pd.read_csv(chains_path, compression="bz2")
                with open(search_path, "rb") as f:
                    search = pickle.load(f)
                planets_fitted = search.post.params.num_planets
                system = universe.systems[system_id]
                system_pops = []
                for i, planet_num in enumerate(range(1, planets_fitted + 1)):
                    planet_chains = self.split_chains(chains, planet_num)
                    system_pops.append(
                        PlanetPopulation(
                            planet_chains,
                            system,
                            self.method,
                            self.n_fits,
                            planet_num,
                        )
                    )
                    system_pops[i].calculate_pdet(self.pdet_times)

                # TEMPORARY PLOTTING
                self.plot(system, system_pops, self.pdet_times)
                # self.pops[system_id] = PlanetPopulation(
                #     planet_chains, system, self.method
                # )
            else:
                logger.warning(f"No chains were created for system {system_id}")

    def split_chains(self, chains, nplan):
        cols_to_rename = ["secosw", "sesinw", "e", "per", "tc", "k"]
        df_cols = [f"{col}{nplan}" for col in cols_to_rename]
        df_cols.append("lnprobability")
        p_chains = chains.loc[:, df_cols]
        return p_chains

    def plot(self, system, system_pops, times):
        mpl.use("Qt5Agg")
        cmap = plt.get_cmap("viridis")
        cvals = np.linspace(0, 1, len(system.planets))
        vector_list = []
        planet_inds = []
        for planet in system.planets:
            # Create all planet position vectors
            # planet_vectors = planet.calc_vectors(times)
            # vector_list.append(planet_vectors)
            vector_list.append(utils.calc_position_vectors(planet, times))

        for ind, t in enumerate(tqdm(times)):
            fig, (ax, pdet_ax) = plt.subplots(ncols=2, figsize=[10, 5])
            # Add inner working angle
            IWA_ang = 0.058 * u.arcsec
            IWA_patch = mpatches.Circle(
                (0, 0),
                IWA_ang.to(u.arcsec).value,
                facecolor="grey",
                edgecolor="black",
                alpha=0.1,
                zorder=5,
            )
            ax.add_patch(IWA_patch)
            # Add the planet populations
            for pop in system_pops:
                planet_inds.append(pop.closest_planet_ind)
                color = cmap(cvals[pop.closest_planet_ind])
                vectors = utils.calc_position_vectors(pop, Time([t.jd], format="jd"))
                ax.scatter(
                    (np.arctan((vectors["x"][0] * u.m) / system.star.dist))
                    .to(u.arcsec)
                    .value,
                    (np.arctan((vectors["y"][0] * u.m) / system.star.dist))
                    .to(u.arcsec)
                    .value,
                    # (vectors["x"][0] * u.m).to(u.AU).value,
                    # (vectors["y"][0] * u.m).to(u.AU).value,
                    alpha=0.75,
                    color=color,
                    s=0.01,
                )
                # pop.pdet[:ind]
                pdet_ax.plot(times.decimalyear[:ind], pop.pdet[:ind], color=color)
            # Add the real planets
            for i, planet in enumerate(system.planets):
                pos = utils.calc_position_vectors(planet, Time([t.jd], format="jd"))
                if i in planet_inds:
                    # Make the planets with orbit fits colorful
                    color = cmap(cvals[i])
                    edge = "k"
                    ms = self.planet_marker_size(
                        pos.z,
                        vector_list[i].z,
                        base_size=5 + planet.radius.to(u.R_earth).value,
                        factor=0.2,
                    )[0]
                else:
                    # Make the others black
                    color = "w"
                    edge = "k"
                    ms = 0.2
                ax.scatter(
                    (np.arctan((pos.x[0] * u.m) / system.star.dist)).to(u.arcsec).value,
                    (np.arctan((pos.y[0] * u.m) / system.star.dist)).to(u.arcsec).value,
                    # (pos.x[0] * u.m).to(u.AU),
                    # (pos.y[0] * u.m).to(u.AU),
                    # planet.vectors.y[ind] * (u.m.to(u.AU)),
                    label=f"Planet {i}",
                    color=color,
                    edgecolor=edge,
                    s=ms,
                )
            ax.set_xlim([-0.4, 0.4])
            ax.set_ylim([-0.4, 0.4])
            ax.set_xlabel("arcsec")
            ax.set_ylabel("arcsec")
            ax.set_title(f"Image plane: {t.decimalyear:.02f} yr")
            pdet_ax.set_xlim([times.decimalyear[0], times.decimalyear[-1]])
            pdet_ax.set_ylim([-0.05, 1.05])
            pdet_ax.set_title("Probability of detection")
            fig.savefig(f"figs/{system.star.name}_{ind:03}.png")
            # if ax.get_subplotspec().is_first_col():
            #     ax.annotate(
            #         "IWA",
            #         xy=(0, 0),
            #         xytext=(0, IWA_ang.to(u.arcsec).value * 1.125),
            #         ha="center",
            #         va="center",
            #         arrowprops=dict(arrowstyle="<-"),
            #         zorder=10,
            #     )

    def planet_marker_size(self, z, all_z, base_size=5, factor=0.5):
        """
        Make the planet marker smaller when the planet is behind the star in its orbit
        """
        z_range = np.abs(max(all_z) - min(all_z))

        # Want being at max z to correspond to a factor of 1 and min z
        # to be a factor of negative 1
        scaled_z = 2 * (z - min(all_z)) / z_range - 1

        marker_size = base_size * (1 + factor * scaled_z)

        return marker_size


class PlanetPopulation:
    """
    Class that holds constructed orbits that could have created the RV data
    """

    def __init__(
        self,
        chains,
        system,
        method,
        n_fits,
        nplan,
        fixed_inc=None,
        fixed_p=0.36,
        fixed_f_sed=None,
    ):
        """
        Args:
            chains (pandas.DataFrame):
                The MCMC chains generated from the orbit fitting process
            system (System):
                The system data
        """

        if method == "multivariate gaussian":
            self.cov_samples = 1000
        self.droppable_cols = ["lnprobability"]
        self.n_fits = n_fits
        self.fixed_inc = fixed_inc
        self.fixed_p = fixed_p
        self.fixed_f_sed = fixed_f_sed

        self.dist_to_star = system.star.dist
        self.Ms = system.star.mass
        self.samples_for_cov = (
            chains.pipe(self.start_pipeline)
            .pipe(self.sort_by_lnprob)
            .pipe(self.get_samples_for_covariance)
            .pipe(self.drop_columns)
        )
        self.cov_df = self.samples_for_cov.cov()
        self.chains_means = (
            chains.pipe(self.start_pipeline).pipe(self.drop_columns).mean()
        )
        chain_samples_np = np.random.multivariate_normal(
            self.chains_means, self.cov_df, size=self.n_fits
        )
        chain_samples = pd.DataFrame(chain_samples_np, columns=self.cov_df.keys())

        # Use those samples and assign the values
        self.T = chain_samples[f"per{nplan}"].to_numpy() * u.d
        self.secosw = chain_samples[f"secosw{nplan}"].to_numpy()
        self.sesinw = chain_samples[f"sesinw{nplan}"].to_numpy()
        self.K = chain_samples[f"k{nplan}"].to_numpy()
        self.T_c = Time(chain_samples[f"tc{nplan}"].to_numpy(), format="jd")

        # Cheating and using the most likely planet's W value instead of random
        # sampling because it doesn't impact imaging, just makes the plots
        # prettier.
        p_df = system.get_p_df()
        planet_vals = np.ones(len(system.planets))
        for key in ["K", "T"]:
            diff = np.abs(p_df[key] - np.median(getattr(self, key)))
            sorted_inds = np.array(diff.sort_values().index)
            for i, ind in enumerate(sorted_inds):
                planet_vals[ind] += i
        self.closest_planet_ind = np.where(planet_vals == np.min(planet_vals))[0][0]
        self.W = np.ones(len(self.T)) * p_df.at[self.closest_planet_ind, "W"] * u.deg

        self.pdet = []

        self.create_population()

    def create_population(self):
        secosw = self.secosw
        sesinw = self.sesinw
        T = self.T
        K = self.K
        T_c = self.T_c
        e = secosw**2 + sesinw**2
        if isinstance(e, np.ndarray):
            e[np.where((e > 1) | (e == 0))] = 0.0001
        else:
            if (e > 1) or (e == 0):
                e = 0.0001
        self.e = e
        # Mass of planet and inclination
        self.Msini = rvu.Msini(K, T, np.ones(T.size) * self.Ms, e, Msini_units="earth")
        if self.fixed_inc is None:
            # Without a fixed inclination the sinusoidal inclination
            # distribution is sampled from and used to calcluate a
            # corresponding mass

            Msini_solar = (self.Msini * u.M_earth).to(u.M_sun)
            if np.any(Msini_solar > 0.08 * u.M_sun):
                # TODO Don't have a good way to handle planets that shouldn't
                # exist this late in execution.
                Icrit = 0
            else:
                Icrit = np.arcsin(
                    (self.Msini * u.M_earth).value
                    / ((0.0800 * u.M_sun).to(u.M_earth)).value
                )
            Irange = [Icrit, np.pi - Icrit]
            C = 0.5 * (np.cos(Irange[0]) - np.cos(Irange[1]))
            inc = np.arccos(
                np.cos(Irange[0]) - 2.0 * C * np.random.uniform(size=len(e))
            )
            self.inc = inc * u.rad
        else:
            # When a fixed inclination is given we don't need to sample anything
            inc = self.fixed_inc
            self.inc = inc
        # Planet mass from inclination
        self.Mp = np.abs(self.Msini / np.sin(inc)) * u.M_earth

        # Use modified FORCASTER model
        self.Rp = self.calc_radius_from_mass(self.Mp)

        # Set geometric albedo
        # if self.fixed_p is not None:
        self.p = self.fixed_p

        # Now with inclination, planet mass, and longitude of the ascending node
        # we can calculate the remaining parameters
        self.mu = (const.G * (self.Mp + self.Ms)).decompose()
        self.a = ((self.mu * (T / (2 * np.pi)) ** 2) ** (1 / 3)).decompose()

        # Finding the planet's arguement of periapsis, RadVel report's the
        # star's value
        self.w_s = np.arctan2(sesinw, secosw) * u.rad
        self.w = (self.w_s + np.pi * u.rad) % (2 * np.pi * u.rad)

        # Finding the mean anomaly at time of conjunction
        nu_p = (np.pi / 2 * u.rad - self.w_s) % (2 * np.pi * u.rad)
        E_p = 2 * np.arctan2(np.sqrt((1 - e)) * np.tan(nu_p / 2), np.sqrt((1 + e)))
        self.M0 = (E_p - e * np.sin(E_p) * u.rad) % (2 * np.pi * u.rad)

        # set initial epoch to the time of conjunction since that's what M0 is
        self.t0 = Time(T_c, format="jd")
        self.T_p = rvo.timetrans_to_timeperi(T_c, self.T, e, self.w.value)

        # Find the semi-amplitude
        self.K = (
            (2 * np.pi * const.G / self.T) ** (1 / 3.0)
            * (self.Mp * np.sin(self.inc) / self.Ms ** (2 / 3.0))
            * (1 - self.e**2) ** (-1 / 2)
        ).decompose()

        # Mean angular motion
        self.n = (np.sqrt(self.mu / self.a**3)).decompose() * u.rad

        if self.fixed_f_sed is None:
            self.f_sed = np.random.choice(
                [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 6],
                self.n_fits,
                p=[0.099, 0.001, 0.005, 0.01, 0.025, 0.28, 0.3, 0.28],
            )
        else:
            self.f_sed = np.ones(self.n_fits) * self.fixed_f_sed
        pass

    def prop_for_imaging(self, t):
        # Calculates the working angle and deltaMag
        a, e, I, w = self.a, self.e, self.inc, self.w
        M = utils.mean_anom(self, t)
        E = kt.eccanom(M.value, self.e)
        nu = kt.trueanom(E, e) * u.rad
        r = a * (1 - e**2) / (1 + e * np.cos(nu))

        theta = nu + w
        s = (r / 4) * np.sqrt(
            4 * np.cos(2 * I)
            + 4 * np.cos(2 * theta)
            - 2 * np.cos(2 * I - 2 * theta)
            - 2 * np.cos(2 * I + 2 * theta)
            + 12
        )
        beta = np.arccos(-np.sin(I) * np.sin(theta))
        # For gas giants
        # p_phi = self.calc_p_phi(beta, photdict, bandinfo)
        # For terrestrial planets
        phi = self.lambert_func(beta)
        # p_phi = self.p * phi
        p_phi = self.p * phi

        WA = np.arctan(s / self.dist_to_star).decompose()
        dMag = -2.5 * np.log10(p_phi * ((self.Rp / r).decompose()) ** 2).value
        return WA, dMag

    def calculate_pdet(self, times):
        IWA = 0.058 * u.arcsec
        OWA = 6 * u.arcsec
        dMag0 = 26.5
        for t in times:
            WA, dMag = self.prop_for_imaging(t)
            visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
            self.pdet.append(sum(visible) / self.n_fits)
        self.pdet = np.array(self.pdet)

    def lambert_func(self, beta):
        return (np.sin(beta) * u.rad + (np.pi * u.rad - beta) * np.cos(beta)) / (
            np.pi * u.rad
        )

    def start_pipeline(self, df):
        return df.copy()

    def get_samples_for_covariance(self, df):
        return df.head(self.cov_samples)

    def sort_by_lnprob(self, df):
        return df.sort_values("lnprobability", ascending=False)

    def drop_columns(self, df):
        return df.drop(columns=self.droppable_cols)

    def calc_radius_from_mass(self, Mp):

        """Calculate planet radius from mass

        Args:
            Mp (astropy Quantity array):
                Planet mass in units of Earth mass

        Returns:
            astropy Quantity array:
                Planet radius in units of Earth radius

        """

        S = np.array([0.2790, 0, 0, 0, 0.881])
        C = np.array([np.log10(1.008), 0, 0, 0, 0])
        T = np.array(
            [
                2.04,
                95.16,
                (u.M_jupiter).to(u.M_earth),
                ((0.0800 * u.M_sun).to(u.M_earth)).value,
            ]
        )

        Rj = u.R_jupiter.to(u.R_earth)
        Rs = 8.522  # saturn radius
        S[1] = (np.log10(Rs) - (C[0] + np.log10(T[0]) * S[0])) / (
            np.log10(T[1]) - np.log10(T[0])
        )
        C[1] = np.log10(Rs) - np.log10(T[1]) * S[1]
        S[2] = (np.log10(Rj) - np.log10(Rs)) / (np.log10(T[2]) - np.log10(T[1]))
        C[2] = np.log10(Rj) - np.log10(T[2]) * S[2]
        C[3] = np.log10(Rj)
        C[4] = np.log10(Rj) - np.log10(T[3]) * S[4]

        m = np.array(Mp.to(u.M_earth).value, ndmin=1)
        R = np.zeros(m.shape)

        # inds = np.digitize(m, np.hstack((0, T, np.inf)))
        # breakpoint()
        # for j in range(1, min(inds.max() + 1), 5):
        #     R[inds == j] = 10.0 ** (C[j - 1] + np.log10(m[inds == j]) * S[j - 1])
        inds = np.digitize(m, np.hstack((0, T, np.inf)))
        for j in range(1, inds.max() + 1):
            R[inds == j] = 10.0 ** (C[j - 1] + np.log10(m[inds == j]) * S[j - 1])

        return R * u.R_earth
