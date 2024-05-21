import json
import pickle
import time
from datetime import datetime
from itertools import repeat
from multiprocessing import Value
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import dill
import erfa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from astropy.time import Time
from keplertools import fun as kt
from pathos.multiprocessing import Pool
from scipy.interpolate import interp1d, interp2d
from tqdm import tqdm

import radvel.orbit as rvo
import radvel.utils as rvu
import RVtoImaging.plots
import RVtoImaging.utils as utils
from EXOSIMS.util.get_module import get_module_from_specs
from EXOSIMS.util.phaseFunctions import realSolarSystemPhaseFunc
from EXOSIMS.util.utils import dictToSortedStr, genHexStr

# from EXOSIMS.util.vprint import vprint
from RVtoImaging.logger import logger


def dim_dMag_obj(fZ, int_times, WAs, fEZs, TL, sInd):
    OS = TL.OpticalSystem
    mode = list(filter(lambda mode: mode["detectionMode"] is True, OS.observingModes))[
        0
    ]
    # Formatting for calc_dMag_per_intTime
    _int_times = [
        np.array(int_time.to(u.d).value, ndmin=1) * u.d for int_time in int_times
    ]
    _WAs = [np.array(WA.to(u.arcsec).value, ndmin=1) * u.arcsec for WA in WAs]
    _fEZs = [np.array(fEZ, ndmin=1) * (1 / u.arcsec**2) for fEZ in fEZs]
    dim_dMag_arr = np.zeros([len(int_times), len(WAs)])
    for i, _int_time in enumerate(_int_times):
        for j, _WA in enumerate(_WAs):
            dim_dMag_arr[i, j] = OS.calc_dMag_per_intTime(
                _int_time,
                TL,
                np.array([sInd]),
                fZ,
                _fEZs[j],
                _WA,
                mode,
            )[0]
            fZ_counter.value += 1
            fZ_pbar.update_to(fZ_counter.value)
    return dim_dMag_arr


class ImagingProbability:
    """
    Base class to do probability of detection calculations
    """

    def __init__(self, params, orbitfit, universe, workers):
        self.method = params["construction_method"]
        self.script_path = Path(params["script"])
        with open(self.script_path) as f:
            specs = json.loads(f.read())
        specs["exoverses_universe"] = universe

        self.SS = get_module_from_specs(specs, "SurveySimulation")(**specs)

        # for sInd in
        # utils.replace_EXOSIMS_system(self.SS, sInd, )
        self.n_fits = params["number_of_orbits"]
        start_time = params["start_time"]
        end_time = params["end_time"]

        min_int_time = params["min_int_time"]
        max_int_time = params["max_int_time"]
        if max_int_time > self.SS.TargetList.OpticalSystem.intCutoff:
            logger.warn("Maximum integration time exceeds EXOSIMS intCutoff")

        include_keepout_in_pdet = params.get("include_keepout_in_pdet", False)
        # max_int_time = self.SS.TargetList.OpticalSystem.intCutoff
        self.int_times = self.gen_int_times(min_int_time, max_int_time)
        self.WAs = self.gen_WAs()
        self.fEZ_quantile = params["fEZ_quantile"]
        self.fEZ_exact = params.get("fEZ_exact", False)

        self.pdet_times = Time(
            np.arange(start_time.jd, end_time.jd, min_int_time.to(u.d).value),
            format="jd",
        )

        self.pops = {}
        self.pdets = {}
        self.mcmc_info = {}
        # tmp = pd.json_normalize(specs, sep=',')
        # items = []
        if not include_keepout_in_pdet:
            self.skipped_EXOSIMS_keys = [
                "koAngles_Sun",
                "koAngles_Earth",
                "koAngles_Moon",
                "koAngles_Small",
                "koAngles_SolarPanel",
            ]
        else:
            self.skipped_EXOSIMS_keys = []
        self.skipped_EXOSIMS_keys.append("missionPortion")
        self.script_hash = utils.EXOSIMS_script_hash(
            self.script_path, skip_list=self.skipped_EXOSIMS_keys
        )
        # for col in tmp.columns:
        #     val = tmp[col]
        settings_str = (
            f"{genHexStr(dictToSortedStr(self.method))}_"
            f"{self.n_fits}_"
            f"{start_time.jd:.2f}_"
            f"{end_time.jd:.2f}_"
            f"{min_int_time.to(u.d).value:.2f}_"
            f"{max_int_time.to(u.d).value:.2f}_"
            f"{self.fEZ_quantile:.2f}_"
            f"{self.script_hash}"
        )
        self.settings_hash = genHexStr(settings_str)
        self.system_paths = {
            universe.systems[system].star.name: path
            for system, path in zip(orbitfit.systems_to_fit, orbitfit.paths)
        }

        self.pdets_to_do = 0
        self.pdets_finished = 0
        # get the number of planets we need to calc pdet for for progress info
        for system_path in orbitfit.paths:
            chains_spec_path = Path(system_path, "spec.json")
            chains_path = Path(system_path, "chains.csv.tar.bz2")
            pdet_path = Path(system_path, f"pdet_{self.settings_hash}.p")
            if pdet_path.exists():
                continue
            if chains_spec_path.exists() and chains_path.exists():
                with open(chains_spec_path, "r") as f:
                    chains_spec = json.loads(f.read())
                if chains_spec["mcmc_converged"]:
                    self.pdets_to_do += 1

        start_time = time.time()
        # Loop through all the systems we want to calculate probability of
        # detection for
        # Check for the chains
        for system_n, (system_id, system_path) in enumerate(
            zip(orbitfit.systems_to_fit, orbitfit.paths)
        ):
            system = universe.systems[system_id]
            if system.star.name.replace("_", " ") not in self.SS.TargetList.Name:
                logger.warning(
                    (
                        f"The {system.star.name} system is not "
                        "in the PDet EXOSIMS TargetList. Continuing..."
                    )
                )
                # continue
            chains_spec_path = Path(system_path, "spec.json")
            chains_path = Path(system_path, "chains.csv.tar.bz2")
            search_path = Path(system_path, "search.pkl")
            # pdet_path = Path(system_path, "pdet.nc")
            pdet_path = Path(system_path, f"pdet_{self.settings_hash}.p")
            pops_path = Path(system_path, f"pops_{self.settings_hash}.p")
            if chains_spec_path.exists():
                with open(chains_spec_path, "r") as f:
                    chains_spec = json.loads(f.read())
                # if not chains_spec["mcmc_converged"]:
                #     logger.warning(
                #         f"Star {system_n} of {len(orbitfit.paths)}. "
                #         f"Skipping {universe.names[system_id]}, chains didn't
                #         converge."
                #     )
                #     continue

            if chains_path.exists():
                if not pdet_path.exists():
                    # if pdet_path.exists():
                    logger.info(
                        (
                            "Loading chains and search data for "
                            f"{universe.names[system_id]}. "
                            f"Star {system_n+1} of {len(orbitfit.paths)}."
                        )
                    )
                    chains = pd.read_csv(chains_path, compression="bz2")
                    with open(search_path, "rb") as f:
                        search = pickle.load(f)
                    planets_fitted = search.post.params.num_planets
                    system = universe.systems[system_id]
                    system_pops = []
                    # Create the integration times and dMag0s
                    # dMag0s = self.gen_dMag0s(self.SS, system, self.int_times)
                    if self.method["name"] == "exact":
                        dim_dMag_interps = None
                    else:
                        dim_dMag_interps = self.gen_dim_dMag_interps(
                            self.SS,
                            system,
                            self.int_times,
                            self.WAs,
                            self.fEZ_quantile,
                            system_path,
                            workers,
                        )
                    system_pdets = pd.DataFrame(
                        np.zeros((len(self.int_times), len(self.pdet_times))),
                        columns=self.pdet_times,
                    )
                    system_pdets.index = self.int_times

                    # Logging info
                    runs_left = self.pdets_to_do - self.pdets_finished
                    if self.pdets_finished > 0:
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        rate = elapsed_time / self.pdets_finished
                        finish_time = datetime.fromtimestamp(
                            current_time + rate * runs_left
                        )
                        finish_str = finish_time.strftime("%c")
                        rate_str = (
                            f"{rate/60:.2f} minutes per "
                            "probability of detection calculation"
                        )
                    else:
                        finish_str = "TBD"
                        rate_str = "Speed unknown"
                    logger.info(
                        f"Star {system_n+1} of {len(orbitfit.paths)}. "
                        f"Calculating probability of detection for {system.star.name}. "
                        f"{rate_str}. "
                        f"Estimated finish: {finish_str}."
                    )

                    input_error = False
                    # Adding progress bar
                    global pbar
                    global counter
                    counter = Value("i", 0, lock=True)
                    pbar = TqdmUpTo(
                        total=len(self.pdet_times) * planets_fitted,
                        desc="Probability of detection",
                    )
                    populations_created = 0
                    for i, planet_num in enumerate(range(1, planets_fitted + 1)):
                        #     tqdm(
                        #         range(1, planets_fitted + 1),
                        #         position=0,
                        #         desc="Fitted planet",
                        #     )
                        # ):
                        if input_error:
                            continue
                        planet_chains = self.split_chains(chains, planet_num)
                        # Check if any system planets are close
                        pop = PlanetPopulation(
                            planet_chains,
                            system,
                            self.method,
                            self.n_fits,
                            planet_num,
                        )
                        # pop.gen_fEZ_corrections(self.SS, system)
                        if pop.input_error is True:
                            input_error = True
                            continue

                        pop.chains_spec = chains_spec
                        system_pops.append(pop)
                        system_pops[i].calculate_pdet(
                            self.pdet_times,
                            self.int_times,
                            dim_dMag_interps,
                            self.SS,
                            workers=workers,
                            ko_included=include_keepout_in_pdet,
                        )
                        system_pdets = system_pdets.add(system_pops[i].pdets)
                        populations_created += 1
                    self.pdets_finished += 1
                    pbar.close()
                    # if input_error:
                    #     logger.warning(
                    #         f"Skipping {universe.names[system_id]},"
                    #         " negative time of conjunction"
                    #     )
                    #     continue

                    self.pops[universe.names[system_id]] = system_pops
                    self.mcmc_info[universe.names[system_id]] = chains_spec
                    pdet_xr = xr.DataArray(
                        np.stack([pop.pdets for pop in system_pops]),
                        dims=["planet", "int_time", "time"],
                        coords=[
                            np.arange(0, populations_created, 1),
                            self.int_times,
                            self.pdet_times.datetime,
                        ],
                    )
                    pdet_xr_set = pdet_xr.to_dataset(name="pdet")
                    # self.plot(system, system_pops, self.pdet_times, system_pdets)
                    # pdet_xr_set.to_netcdf(pdet_path)
                    with open(pdet_path, "wb") as f:
                        dill.dump(pdet_xr_set, f)
                    with open(pops_path, "wb") as f:
                        dill.dump(system_pops, f)
                else:
                    logger.info(
                        f"Star {system_n+1} of {len(orbitfit.paths)}. "
                        f"Probability of detection already exists for"
                        f" {universe.systems[system_id].star.name}. Loading..."
                    )
                    # pdet_xr_set = xr.load_dataset(
                    #     pdet_path, decode_cf=True, decode_times=True, engine="netcdf4"
                    # )
                    with open(pdet_path, "rb") as f:
                        pdet_xr_set = dill.load(f)
                    with open(pops_path, "rb") as f:
                        self.pops[universe.names[system_id]] = dill.load(f)

                    system_pops = self.pops[universe.names[system_id]]
                    if not hasattr(system_pops[0], "chains_spec"):
                        rvdf = pd.read_csv(Path(system_path.parent, "rv.csv"))
                        chains_spec["best_precision"] = rvdf.errvel.min()

                        for pop in system_pops:
                            pop.chains_spec = chains_spec
                        with open(pops_path, "wb") as f:
                            pickle.dump(system_pops, f)
                        utils.update(chains_spec_path.parent, chains_spec)
                        self.pops[universe.names[system_id]] = system_pops
                    # self.pops[universe.names[system_id]] = system_pops

                # TEMPORARY PLOTTING
                # self.pops[system_id] = PlanetPopulation(
                #     planet_chains, system, self.method
                # )
                self.pdets[system.star.name] = pdet_xr_set
            else:
                logger.warning(
                    f"Star {system_n} of {len(orbitfit.paths)}. "
                    f"No chains were created for system {system_id}. "
                )

    def gen_dMag0s(self, SS, system, int_times):
        TL = SS.TargetList
        OS = TL.OpticalSystem
        ZL = TL.ZodiacalLight
        IWA = OS.IWA
        OWA = OS.OWA
        mid_WA = (
            np.array([(IWA.to(u.arcsec).value + OWA.to(u.arcsec).value) / 2]) * u.arcsec
        )
        fZ = ZL.fZ0
        fEZ = ZL.fEZ0
        mode = list(
            filter(lambda mode: mode["detectionMode"] is True, OS.observingModes)
        )[0]

        target_sInd = np.where(TL.Name == system.star.name.replace("_", " "))[0]
        # except:
        #     target_sInd = np.where(
        #         SS.StarCatalog.Name == system.star.name.replace("_", " ")
        #     )[0][0]
        dMag0s = []
        # Getting correct array format
        fZ = np.array(fZ.to(1 / u.arcsec**2).value, ndmin=1) * (1 / u.arcsec**2)
        fEZ = np.array(fEZ.to(1 / u.arcsec**2).value, ndmin=1) * (1 / u.arcsec**2)
        for int_time in int_times:
            _int_time = np.array(int_time.to(u.d).value, ndmin=1) * u.d
            dMag0s.append(
                OS.calc_dMag_per_intTime(
                    _int_time, TL, target_sInd, fZ, fEZ, mid_WA, mode
                )[0]
            )
        return dMag0s

    def gen_dim_dMag_interps(
        self, SS, system, int_times, WAs, fEZ_quantile, system_path, workers
    ):
        TL = SS.TargetList
        OS = TL.OpticalSystem
        ZL = TL.ZodiacalLight
        fZ = ZL.fZ0
        # fEZ0 = ZL.fEZ0

        mode = list(
            filter(lambda mode: mode["detectionMode"] is True, OS.observingModes)
        )[0]
        target_sInd = np.where(TL.Name == system.star.name.replace("_", " "))[0]
        fZvals = ZL.fZMap[mode["systName"]][target_sInd][0]
        max_fZ = max(fZvals)
        min_fZ = min(fZvals)
        fZs = np.logspace(0, np.log10(max_fZ / min_fZ), 10) * min_fZ

        with open(self.script_path, "r") as f:
            specs = json.load(f)
        if "seed" in specs.keys() and not self.fEZ_exact:
            specs.pop("seed")
        hash = utils.EXOSIMS_script_hash(
            None, specs=specs, skip_list=self.skipped_EXOSIMS_keys
        )

        d = WAs.to(u.rad).value * TL.dist[target_sInd].to(u.AU)
        inc = np.repeat(135, len(WAs)) * u.deg
        MV = TL.MV[target_sInd]
        # Absolute magnitude of the star (in the V band)
        MV = np.array(MV, ndmin=1, copy=False)
        # Absolute magnitude of the Sun (in the V band)
        MVsun = 4.83

        # # Generating a nEZ value
        # nStars = len(MV)
        if self.fEZ_exact:
            # fEZ = SS.SimulatedUniverse.fEZ
            nEZ = ZL.nEZ_star[target_sInd][0]
            # fudge factor to account for different inclinations
            nEZ += 0.5
            nEZ_str = f"exact_{nEZ:.2f}.p"
        else:
            nEZ_arr = ZL.gen_systemnEZ(10000)
            nEZ = np.quantile(nEZ_arr, fEZ_quantile)
            nEZ_str = f"quant_{fEZ_quantile:.2f}.p"

        # inclinations should be strictly in [0, pi], but allow for weird sampling:
        beta = inc.to("deg").value
        beta[beta > 180] -= 180

        # latitudinal variations are symmetric about 90 degrees so compute the
        # supplementary angle for inclination > 90 degrees
        mask = beta > 90
        beta[mask] = 180.0 - beta[mask]

        # finally, the input to the model is 90-inclination
        beta = 90.0 - beta
        fbeta = ZL.zodi_latitudinal_correction_factor(beta * u.deg, model="interp")

        fEZs = (
            nEZ
            * 10 ** (-0.4 * ZL.magEZ)
            * 10.0 ** (-0.4 * (MV - MVsun))
            * fbeta
            / d.to("AU").value ** 2
            / u.arcsec**2
            * 1
        )

        interps_path = Path(
            system_path.parents[3],
            "dMag_interps",
            hash,
            (
                f"{system.star.name.replace(' ', '_')}_{genHexStr(str(fZs))}_"
                f"{nEZ_str}"
            ),
        )
        interps_path.parent.mkdir(parents=True, exist_ok=True)
        if interps_path.exists():
            logger.info(f"Loading dimmest dMag interpolants from {interps_path}")
            with open(interps_path, "rb") as f:
                dim_dMag_interps = pickle.load(f)
        else:
            logger.info(f"Creating dimmest dMag interpolants for {system.star.name}")

            global fZ_pbar
            global fZ_counter
            fZ_pbar = TqdmUpTo(
                total=len(int_times) * len(WAs) * len(fZs),
                desc=(
                    f"dMag curves, "
                    f"int times:{len(int_times)}, "
                    f"WAs:{len(WAs)}, "
                    f"fZs:{len(fZs)}"
                ),
            )
            fZ_counter = Value("i", 0, lock=True)

            # Formatting for calc_dMag_per_intTime
            _fZs = [np.array(fZ, ndmin=1) * (1 / u.arcsec**2) for fZ in fZs]
            func_args = zip(
                _fZs,
                repeat(int_times),
                repeat(WAs),
                repeat(fEZs),
                repeat(TL),
                repeat(target_sInd),
            )

            with Pool(processes=workers) as pool:
                result = pool.starmap(dim_dMag_obj, func_args)

            dim_dMag_interps = {}
            for fZ, arr in zip(fZs, result):
                dim_dMag_interps[fZ] = interp2d(
                    WAs.to(u.arcsec).value,
                    int_times.to(u.d).value,
                    arr,
                    fill_value=np.nan,
                )

            with open(interps_path, "wb") as f:
                pickle.dump(dim_dMag_interps, f)
            logger.info(f"Dimmest dMag interpolants saved to {interps_path}")
        return dim_dMag_interps

    def gen_int_times(self, min_time, max_time):
        # Setting up array of integration times
        maxminratio = (max_time / min_time).decompose().value
        base = 2
        maxn = int(round(np.emath.logn(base, maxminratio), 0))
        tmp = [(base**n) for n in range(0, maxn + 1)]
        tmp2 = [tmp[i] + tmp[i - 1] for i in range(1, len(tmp) - 1)]
        combined = np.sort(np.array(tmp + tmp2))
        filtered = [x for x in combined if x <= maxminratio]
        filtered.append(maxminratio)
        int_times = min_time.to(u.d) * filtered
        return int_times

    def gen_WAs(self):
        OS = self.SS.TargetList.OpticalSystem
        mode = list(
            filter(lambda mode: mode["detectionMode"] is True, OS.observingModes)
        )[0]
        IWA = mode["IWA"]
        OWA = mode["OWA"]
        WAs = np.logspace(0, np.log10(OWA / IWA).value, 10) * IWA
        return WAs

    def split_chains(self, chains, nplan):
        cols_to_rename = ["secosw", "sesinw", "e", "per", "tc", "k"]
        df_cols = [f"{col}{nplan}" for col in cols_to_rename]
        df_cols.append("lnprobability")
        p_chains = chains.loc[:, df_cols]
        return p_chains

    def plot(self, system, system_pops, times):
        mpl.use("Qt5Agg")
        cmap = plt.get_cmap("viridis")
        cvals = np.linspace(0, 1, len(system_pops))
        vector_list = []
        for planet in system.planets:
            # Create all planet position vectors
            # planet_vectors = planet.calc_vectors(times)
            # vector_list.append(planet_vectors)
            vector_list.append(utils.calc_position_vectors(planet, times))

        azim_range = np.linspace(15, 75, len(times))
        elev_range = np.linspace(20, 30, len(times))
        roll_range = np.linspace(0, 10, len(times))
        for ind, t in enumerate(tqdm(times)):
            planet_inds = []
            color_mapping = {}
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            ax.view_init(elev_range[0], azim_range[ind], roll_range[0])
            pdet_ax = fig.add_subplot(1, 2, 2)
            current_cmap_ind = 0
            # Add the planet populations
            for pop in system_pops:
                planet_inds.append(pop.closest_planet_ind)
                color = cmap(cvals[current_cmap_ind])
                color_mapping[pop.closest_planet_ind] = color
                current_cmap_ind += 1
                ax = RVtoImaging.plots.pop_3d(ax, pop, t.jd, color)
            pdet_ax.plot(times.decimalyear[:ind], pop.pdets[:ind], color=color)
            # Add the real planets
            for i, planet in enumerate(system.planets):
                pos = utils.calc_position_vectors(planet, Time([t.jd], format="jd"))
                if i in planet_inds:
                    # Make the planets with orbit fits colorful
                    color = color_mapping[i]
                    # color = cmap(cvals[i])
                    edge = "k"
                    # ms = self.planet_marker_size(
                    #     pos.z,
                    #     vector_list[i].z,
                    #     base_size=5 + planet.radius.to(u.R_earth).value,
                    #     factor=0.2,
                    # )[0]
                    ms = 5
                else:
                    # Make the others black
                    color = "w"
                    edge = "k"
                    ms = 0.2
                x = (np.arctan((pos.x[0] * u.m) / system.star.dist)).to(u.arcsec).value
                y = (np.arctan((pos.y[0] * u.m) / system.star.dist)).to(u.arcsec).value
                z = (np.arctan((pos.z[0] * u.m) / system.star.dist)).to(u.arcsec).value
                ax.scatter(
                    x,
                    y,
                    z,
                    # (pos.x[0] * u.m).to(u.AU),
                    # (pos.y[0] * u.m).to(u.AU),
                    # planet.vectors.y[ind] * (u.m.to(u.AU)),
                    label=f"Planet {i}",
                    color=color,
                    edgecolor=edge,
                    s=ms,
                )
                # ax.plot3D(x, y, z, color=color)
            ax.set_xlim([-2.5, 2.5])
            ax.set_ylim([-2.5, 2.5])
            ax.set_zlim([-2.5, 2.5])
            ax.set_xlabel("x arcsec")
            ax.set_ylabel("y arcsec")
            ax.set_zlabel("z arcsec")
            ax.set_title(f"Time: {t.decimalyear:.02f}")
            pdet_ax.set_xlim([times.decimalyear[0], times.decimalyear[-1]])
            pdet_ax.set_ylim([-0.05, 1.05])
            pdet_ax.set_title("Probability of detection")
            fig.savefig(
                f"figs/{system.star.name}_{ind:03}.png",
                dpi=300,
                facecolor="white",
                transparent=False,
            )
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
        fixed_p=0.2,
        fixed_f_sed=None,
    ):
        """
        Args:
            chains (pandas.DataFrame):
                The MCMC chains generated from the orbit fitting process
            system (System):
                The system data
        """
        # WARNING: Remove this
        # self.system = system

        self.method = method
        self.method_name = method["name"]
        self.n_fits = n_fits
        self.fixed_inc = fixed_inc
        self.fixed_p = fixed_p
        self.fixed_f_sed = fixed_f_sed

        # Create the fitting basis params based on the desired method
        if self.method_name == "multivariate gaussian":
            self.cov_samples = method["cov_samples"]
            self.setup_mg(chains, nplan)
        elif self.method_name == "credible interval":
            self.setup_ci(chains, nplan)

        self.dist_to_star = system.star.dist
        self.Ms = system.star.mass
        self.star = system.star
        # self.id = system.star.id
        self.name = system.star.name.replace("_", " ")

        if self.method_name == "exact":
            self.setup_exact(chains, system, nplan)
            self.input_error = False
            return

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
        if np.any(np.sign(self.T_c.jd) == -1):
            # Error check
            self.input_error = True
        elif np.any(self.T.to(u.yr).value > 1e7):
            self.input_error = True
        else:
            self.closest_planet_ind = np.where(planet_vals == np.min(planet_vals))[0][0]
            self.W = (
                np.ones(len(self.T)) * p_df.at[self.closest_planet_ind, "W"] * u.deg
            )
            self.input_error = False
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
        self.circular_mask = e < 0.01
        if np.mean(self.circular_mask) > 0.5:
            # logger.info("Assuming planet is on a circular orbit")
            self.circular = True
            self.e = np.zeros(self.n_fits)
            self.w_s = np.zeros(self.n_fits) * u.rad
            self.w = np.zeros(self.n_fits) * u.rad
            self.M0 = np.pi / 2 * u.rad
        else:
            self.circular = False
            self.e = e
            # Finding the planet's arguement of periapsis, RadVel report's the
            # star's value
            self.w_s = np.arctan2(sesinw, secosw) * u.rad
            self.w = (self.w_s + np.pi * u.rad) % (2 * np.pi * u.rad)
            # Finding the mean anomaly at time of conjunction
            nu_p = (3 * np.pi / 2 * u.rad - self.w_s) % (2 * np.pi * u.rad)
            E_p = 2 * np.arctan2(np.sqrt((1 - e)) * np.tan(nu_p / 2), np.sqrt((1 + e)))
            self.M0 = (E_p - e * np.sin(E_p) * u.rad) % (2 * np.pi * u.rad)

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

        # Limit the size to Jupiter mass
        self.Mp[np.where(self.Mp > 1 * u.M_jupiter)[0]] = 1 * u.M_jupiter

        # Use modified FORCASTER model
        self.Rp = self.calc_radius_from_mass(self.Mp)

        # Set geometric albedo
        # if self.fixed_p is not None:
        # self.p = self.fixed_p

        # Now with inclination, planet mass, and longitude of the ascending node
        # we can calculate the remaining parameters
        self.mu = (const.G * (self.Mp + self.Ms)).decompose()
        self.a = ((self.mu * (T / (2 * np.pi)) ** 2) ** (1 / 3)).decompose()

        # Classify the planets, assigns p
        self.classify_planets()

        # set initial epoch to the time of conjunction since that's what M0 is
        # if self.circular:
        #     self.t0 = Time(T_c, format="jd") - 3 * self.T / 4
        # else:
        self.t0 = Time(T_c, format="jd")
        try:
            self.T_p = rvo.timetrans_to_timeperi(T_c, self.T, e, self.w.value)
        except erfa.core.ErfaError:
            # This occurs for poor fits that get values before the first Julian day
            self.input_error = True

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

    def setup_mg(self, chains, nplan):
        self.droppable_cols = ["lnprobability"]
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

        # For poor fits with high eccentricity the mean/std can result in
        # negative values. This sets negative values to the mean value instead
        # of redrawing.
        if np.any(self.T < 0):
            self.T[np.where(self.T < 0)] = self.chains_means[f"per{nplan}"] * u.d

    def setup_ci(self, chains, nplan):
        quantiles = chains.quantile(0.5)
        std = chains.std()
        self.T = (
            np.random.normal(
                quantiles[f"per{nplan}"], std[f"per{nplan}"], size=self.n_fits
            )
            * u.d
        )
        self.secosw = np.random.normal(
            quantiles[f"secosw{nplan}"], std[f"secosw{nplan}"], size=self.n_fits
        )
        self.sesinw = np.random.normal(
            quantiles[f"sesinw{nplan}"], std[f"sesinw{nplan}"], size=self.n_fits
        )
        self.K = (
            np.random.normal(quantiles[f"k{nplan}"], std[f"k{nplan}"], size=self.n_fits)
            * u.m
            / u.s
        )
        self.T_c = Time(
            np.random.normal(
                quantiles[f"tc{nplan}"], std[f"tc{nplan}"], size=self.n_fits
            ),
            format="jd",
        )

    def setup_exact(self, chains, system, nplan):
        p_df = system.get_p_df()
        planet_vals = np.ones(len(system.planets))
        chain_key_map = {"K": f"k{nplan}", "T": f"per{nplan}"}
        for key in ["K", "T"]:
            diff = np.abs(p_df[key] - chains[chain_key_map[key]].quantile(0.5))
            sorted_inds = np.array(diff.sort_values().index)
            for i, ind in enumerate(sorted_inds):
                planet_vals[ind] += i
        closest_planet_ind = np.where(planet_vals == np.min(planet_vals))[0][0]
        self.T = system.planets[closest_planet_ind].T.reshape(1)
        self.secosw = system.planets[closest_planet_ind].secosw.reshape(1)
        self.sesinw = system.planets[closest_planet_ind].sesinw.reshape(1)
        self.K = system.planets[closest_planet_ind].K.reshape(1)
        self.T_c = system.planets[closest_planet_ind].T_c.reshape(1)
        self.w = system.planets[closest_planet_ind].w.reshape(1)
        self.W = system.planets[closest_planet_ind].W.reshape(1)
        self.inc = system.planets[closest_planet_ind].inc.reshape(1)
        self.a = system.planets[closest_planet_ind].a.reshape(1)
        self.e = np.array([system.planets[closest_planet_ind].e])
        self.T_p = system.planets[closest_planet_ind].T_p.reshape(1)
        self.n = system.planets[closest_planet_ind].n.reshape(1)
        self.t0 = self.T_c
        self.mu = system.planets[closest_planet_ind].mu.reshape(1)
        self.M0 = utils.mean_anom(system.planets[closest_planet_ind], self.t0)
        self.Rp = system.planets[closest_planet_ind].radius.reshape(1)
        self.Mp = system.planets[closest_planet_ind].mass.reshape(1)
        self.circular = self.e == 0
        self.phiIndex = np.array([system.planets[closest_planet_ind].exosims_phiIndex])
        Rp_p_vals = [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5]
        self.p = np.array([Rp_p_vals[self.phiIndex[0]]])

    def prop_for_imaging(self, t):
        # Calculates the working angle and deltaMag
        a, e, inc, w = self.a, self.e, self.inc, self.w
        if self.circular:
            theta = utils.mean_anom(self, t)
            # theta = (math.tau * (phase - np.floor(phase))) * u.rad
            r = a
        else:
            M = utils.mean_anom(self, t)
            E = kt.eccanom(M.value, e)
            nu = kt.trueanom(E, e) * u.rad
            theta = nu + w
            r = a * (1 - e**2) / (1 + e * np.cos(nu))

        s = r * np.sqrt(1 - np.sin(inc) ** 2 * np.sin(theta) ** 2)
        beta = np.arccos(-np.sin(inc) * np.sin(theta))
        phi = realSolarSystemPhaseFunc(beta, self.phiIndex)

        WA = np.arctan(s / self.dist_to_star).decompose()
        dMag = -2.5 * np.log10(self.p * phi * ((self.Rp / r).decompose()) ** 2).value

        return WA, dMag

    def calculate_pdet(
        self, obs_times, int_times, dim_dMag_interps, SS, workers=1, ko_included=False
    ):
        if self.method_name == "exact":
            SU = SS.SimulatedUniverse
            TL = SS.TargetList
            OS = TL.OpticalSystem
            ZL = SS.ZodiacalLight
            TK = SS.TimeKeeping
            Obs = SS.Observatory
            self.t0 = TK.currentTimeAbs.copy()
            self.M0 = utils.mean_anom(self, self.t0)
            mode = list(
                filter(lambda mode: mode["detectionMode"] is True, OS.observingModes)
            )[0]
            # WARNING: I think this is robust when the method is exact, but
            # technically it's possible for two planets to have the exact same
            # period
            pInd = np.where(SS.SimulatedUniverse.a == self.a)[0][0]
            sInd = SU.plan2star[pInd]
            # Set up loop for each observation time
            pdets = np.zeros((len(int_times), len(obs_times)))
            dt = (obs_times[1].jd - obs_times[0].jd) * u.d
            _sInd = np.array([sInd])
            _fEZ = SU.fEZ[pInd].reshape(1)
            # Cps = np.zeros(len(obs_times))
            # Cbs = np.zeros(len(obs_times))
            # Csps = np.zeros(len(obs_times))
            Cps = []
            Cbs = []
            Csps = []
            # Generate all the count rates
            for i, obs_time in enumerate(obs_times):
                # Propagate the planet to the observation time
                SU.propag_system(sInd, dt)
                WA = SU.WA[pInd].reshape(1)
                dMag = SU.dMag[pInd].reshape(1)
                fZ = ZL.fZ(Obs, TL, sInd, obs_time, mode)[0].reshape(1)
                Cp, Cb, Csp = OS.Cp_Cb_Csp(TL, _sInd, fZ, _fEZ, dMag, WA, mode, TK=TK)
                Cps.append(Cp[0])
                Cbs.append(Cb[0])
                Csps.append(Csp[0])
                # Cps[i], Cbs[i], Csps[i] = Cp[0], Cb[0], Csp[0]
                counter.value += 1
                pbar.update_to(counter.value)
            # Get the count rates for the correct time which should be halfway
            # through the observation window
            Cps = u.Quantity(Cps)
            Cbs = u.Quantity(Cbs)
            Csps = u.Quantity(Csps)
            for i, int_time in enumerate(int_times):
                # Start by getting the correct obs time index offset for the
                # integration time
                offset = int((int_time.to(u.d).value / 2) / dt.to(u.d).value)
                if offset < 1:
                    # If the offset is less than the time between observations
                    # we can just use the count rates from original obs_time
                    map = np.full(len(obs_times), True, dtype=bool)
                    signal = (Cps * int_time).decompose().value
                    noise = np.sqrt(
                        (Cbs * int_time + (Csps * int_time) ** 2).decompose().value
                    )
                else:
                    map = np.full(len(obs_times), True, dtype=bool)
                    map[:offset] = False
                    signal = np.zeros(len(obs_times))
                    noise = np.zeros(len(obs_times))
                    signal[:-offset] = (Cps[map] * int_time).decompose().value
                    noise[:-offset] = np.sqrt(
                        (Cbs[map] * int_time + (Csps[map] * int_time) ** 2)
                        .decompose()
                        .value
                    )

                SNR = signal / noise
                obs_time_pdets = SNR >= (mode["SNR"] + 0.1)
                pdets[i, :] = obs_time_pdets

            self.pdets = pd.DataFrame(pdets, columns=obs_times)
            SS.reset_sim(genNewPlanets=False)
        else:
            # Unpacking necessary values
            TL = SS.TargetList
            OS = TL.OpticalSystem
            ZL = SS.ZodiacalLight
            IWA = OS.IWA
            OWA = OS.OWA
            mode = list(
                filter(lambda mode: mode["detectionMode"] is True, OS.observingModes)
            )[0]
            target_sInd = np.where(TL.Name == self.name)[0][0]
            target_ko = SS.koMaps[mode["syst"]["name"]][target_sInd]
            koT = SS.koTimes
            if not ko_included:
                target_ko = np.ones(len(koT), dtype=bool)

            # self.prop_for_imaging(obs_times[0])
            # Calculating the WA and dMag values of all orbits at all times
            with Pool(processes=workers) as pool:
                output = pool.map(self.WA_dMag_obj, obs_times)
            all_WAs = []
            all_dMags = []
            for chunk in output:
                all_WAs.append(chunk[0].to(u.arcsec).value)
                all_dMags.append(chunk[1])
            WAarr = np.array(all_WAs)
            dMagarr = np.array(all_dMags)

            fZ_interp = interp1d(koT.jd, ZL.fZMap[mode["systName"]][target_sInd])
            # WARNING: This should probably raise an error instead of clipping
            # the values.
            # For now I think this only gets this far when the "length" of the
            # schedule and the mission length are the same, but account for
            # leap years differently
            obs_times_jd = obs_times.jd
            obs_times_jd[obs_times_jd > koT.jd[-1]] = koT.jd[-1]
            fZs = fZ_interp(obs_times_jd)
            fZ_interp_keys = np.array([x for x in dim_dMag_interps.keys()])
            fZ_time_keys = np.zeros(len(fZs))
            for i, fZ in enumerate(fZs):
                greater_fZs = fZ_interp_keys[fZ_interp_keys > fZ]
                if len(greater_fZs) == 0:
                    fZ_time_keys[i] = fZ_interp_keys[-1]
                else:
                    fZ_time_keys[i] = min(greater_fZs)

            fZs = fZs * (1 / u.arcsec**2)
            func_args = zip(
                obs_times.jd,
                WAarr,
                dMagarr,
                fZ_time_keys,
                repeat(koT.jd),
                repeat(target_ko),
                repeat(IWA.to(u.arcsec).value),
                repeat(OWA.to(u.arcsec).value),
                repeat(sorted(int_times.to(u.d).value)),
                repeat(dim_dMag_interps),
            )
            with Pool(processes=workers) as pool:
                output = pool.starmap(self.pdet_obj, func_args)

            # Turn output list of lists into a dataframe
            output_arr = np.array(output)
            self.pdets = pd.DataFrame(output_arr.T, columns=obs_times)
            self.pdets.index = int_times

    def WA_dMag_obj(self, obs_time):
        WA, dMag = self.prop_for_imaging(obs_time)
        return (WA, dMag)

    def pdet_obj(
        self,
        obs_time,
        WA,
        dMag,
        fZ_key,
        koT,
        target_ko,
        IWA,
        OWA,
        int_times,
        dim_dMag_interps,
    ):
        counter.value += 1
        pbar.update_to(counter.value)
        pdets = np.zeros(len(int_times))

        # Find the day of the observation for the keepout map
        starts_in_keepout = target_ko[koT <= obs_time][-1]
        if not starts_in_keepout:
            # If the observation starts in keepout, ignore this observation time
            # by returning zeros
            return pdets

        # Mask with obscuration constraint
        obscuration_mask = (IWA < WA) & (OWA > WA)

        # This mask gets updated as more planets meet the threshold
        worth_checking_mask = obscuration_mask
        worth_checking_inds = np.where(worth_checking_mask == 1)[0]
        known_visible = 0

        after_current_time = koT >= obs_time
        interp_fun = dim_dMag_interps[fZ_key]

        for i, int_time in enumerate(int_times):
            before_final_time = koT <= (obs_time + int_time)
            relevant_ko = target_ko[before_final_time & after_current_time]

            if np.any(~relevant_ko):
                # If any part of the observation would be in keepout,
                # ignore this observation time
                break
            else:
                if worth_checking_inds.size > 0:
                    visible_for_int_time = (
                        interp_fun(WA[worth_checking_inds], int_time)
                    ) > dMag[worth_checking_inds]
                    known_visible += np.sum(visible_for_int_time)
                    worth_checking_inds = worth_checking_inds[~visible_for_int_time]
                pdets[i] = known_visible / self.n_fits

                # Remove from checking because it will be detectable for a
                # higher integration time by default
        return pdets

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
        """
        Calculate planet radius from mass, stolen from EXOSIMS

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

        inds = np.digitize(m, np.hstack((0, T, np.inf)))
        # Catch, can't handle larger masses
        inds[inds > 4] = 4
        for j in range(1, inds.max() + 1):
            R[inds == j] = 10.0 ** (C[j - 1] + np.log10(m[inds == j]) * S[j - 1])

        return R * u.R_earth

    def classify_planets(self):
        """
        This determines the Kopparapu bin of the planet This is adapted from
        the EXOSIMS SubtypeCompleteness method classifyPlanets so that EXOSIMS
        isn't a mandatory import
        """
        # Calculate the luminosity of the star, assuming main-sequence
        if self.star.mass < 2 * u.M_sun:
            self.Ls = const.L_sun * (self.star.mass / const.M_sun) ** 4
        else:
            self.Ls = 1.4 * const.L_sun * (self.star.mass / const.M_sun) ** 3.5

        a_AU = self.a.to(u.AU).value
        Rp_earth = self.Rp.to("earthRad").value
        self.phiIndex = np.zeros(len(a_AU))
        self.p = np.zeros(len(a_AU))
        for i, a, e, Rp in zip(range(len(a_AU)), a_AU, self.e, Rp_earth):
            # Find the stellar flux at the planet's location as a fraction of earth's
            earth_Lp = const.L_sun / (1 * (1 + (0.0167**2) / 2)) ** 2
            self.Lp = self.Ls / (a * (1 + (e**2) / 2)) ** 2 / earth_Lp

            # Find Planet Rp range
            Rp_bins = np.array([0, 0.5, 1.0, 1.75, 3.5, 6.0, 14.3, 11.2 * 4.6])
            Rp_types = [
                "Sub-Rocky",
                "Rocky",
                "Super-Earth",
                "Sub-Neptune",
                "Sub-Jovian",
                "Jovian",
                "Super-Jovian",
            ]
            Rp_phase_inds = [0, 2, 2, 7, 7, 4, 4]
            Rp_p_vals = [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5]
            L_bins = np.array(
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

            # Find the bin of the radius
            Rp_bin = np.digitize(Rp, Rp_bins) - 1
            Rp_bin = max(0, min(Rp_bin, len(Rp_types) - 1))

            # Set the albedo and phase index
            self.Rp_type = Rp_types[Rp_bin]
            self.phiIndex[i] = Rp_phase_inds[Rp_bin]
            self.p[i] = Rp_p_vals[Rp_bin]

            # TODO Fix this to give correct when at edge cases since technically
            # they're not straight lines

            # index of planet temp. cold,warm,hot
            L_types = ["Very Hot", "Hot", "Warm", "Cold", "Very Cold"]
            specific_L_bins = L_bins[Rp_bin, :]
            L_bin = np.digitize(self.Lp.decompose().value, specific_L_bins) - 1
            L_bin = max(0, min(L_bin, len(L_types) - 1))
            self.L_type = L_types[L_bin]


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize
