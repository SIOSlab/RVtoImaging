import copy
import itertools
import json
import subprocess
from collections.abc import MutableMapping
from pathlib import Path

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import pandas as pd
from keplertools import fun as kt
from tqdm import tqdm

from EXOSIMS.util.phaseFunctions import realSolarSystemPhaseFunc
from EXOSIMS.util.utils import dictToSortedStr, genHexStr
from RVtoImaging.logger import logger


def runcmd(cmd, verbose=False):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def get_data(universes=np.arange(1, 13), cache_location="data"):
    """
    This function gets all the exoVista data. It gets the csv file with the universe
    information and puts it in the "data/{universe_number}/target_database.csv".
    Then it goes through every url in the csv file and pulls that fits file into
    "data/{universe_number}/{file}.fits".
    """
    # Iterate over the different universes
    for n in tqdm(universes, position=0, desc="Universe", leave=False):
        universe_url = (
            "https://ckan-files.emac.gsfc.nasa.gov/"
            f"exovista/DEC21/{n}/target_database.csv"
        )
        Path(cache_location, str(n)).mkdir(parents=True, exist_ok=True)
        if not Path(cache_location, str(n), "target_database.csv").exists():
            runcmd(f"wget --directory-prefix=data/{n} {universe_url}", verbose=False)

        df = pd.read_csv(
            Path(cache_location, str(n), "target_database.csv"), low_memory=False
        )
        for i in tqdm(
            np.arange(1, df.shape[0]), position=1, desc="System", leave=False
        ):
            # Get file url
            fit_url = df.at[i, "URL"]

            # Create file path
            file_path = Path(cache_location, str(n), f"{fit_url.split('/')[-1]}")

            # If file doesn't exist then pull it
            if not file_path.exists():
                runcmd(
                    f"wget --directory-prefix=data/{n} {fit_url}",
                    verbose=False,
                )

            # Verify data and repull it if necessary
            pull_failure = True
            while pull_failure:
                pull_failure = check_data(file_path)
                if pull_failure:
                    runcmd(
                        f"wget --directory-prefix=data/{n} {fit_url}",
                        verbose=False,
                    )


def check_data(file):
    """
    This function verifies that all the fits files have the necessary data, sometimes
    they don't pull everything for some reason
    """
    system_info = fits.info(file, output=False)
    with open(file, "rb") as f:
        # read header of first extension
        h = fits.getheader(f, ext=0, memmap=False)
    n_ext = h["N_EXT"]  # get the largest extension
    failure = len(system_info) != n_ext + 1
    if failure:
        # Number of tables doesn't match the number of tables that the header
        # says exists, delete file
        file.unlink()
    return failure


def mean_anom(obj, times):
    """
    Calculate the mean anomaly at the given times
    Args:
        times (astropy Time array):
            Times to calculate mean anomaly

    Returns:
        M (astropy Quantity array):
            Planet's mean anomaly at t (radians)
    """
    M = ((obj.n * ((times.jd - obj.t0.jd) * u.d)).decompose() + obj.M0) % (
        2 * np.pi * u.rad
    )
    return M


def calc_position_vectors(obj, times, unit=None):
    orbElem = (
        np.array(obj.a.decompose().value, ndmin=1),
        np.array(obj.e, ndmin=1),
        np.array(obj.W.to(u.rad).value, ndmin=1),
        np.array(obj.inc.to(u.rad).value, ndmin=1),
        np.array(obj.w.to(u.rad).value, ndmin=1),
    )
    x, y, z, vx, vy, vz = [], [], [], [], [], []
    for t in times:
        M = mean_anom(obj, t)
        E = kt.eccanom(M.to(u.rad).value, obj.e)
        mu = obj.mu.decompose().value
        r, v = kt.orbElem2vec(E, mu, orbElem)
        if unit is not None:
            r = (r * u.m).to(unit).value
        x.append(r[0])
        y.append(r[1])
        z.append(r[2])
        vx.append(v[0])
        vy.append(v[1])
        vz.append(v[2])
    r = {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz, "t": times.jd}
    full_vector = pd.DataFrame(r)
    return full_vector


def calc_position_vectors2(obj, times):
    orbElem = (
        np.array(obj.a.decompose().value, ndmin=1),
        np.array(obj.e, ndmin=1),
        np.array(obj.W.to(u.rad).value, ndmin=1),
        np.array(obj.inc.to(u.rad).value, ndmin=1),
        np.array(obj.w.to(u.rad).value, ndmin=1),
    )
    x, y, z, vx, vy, vz = [], [], [], [], [], []
    for t in times:
        M = mean_anom(obj, t)
        E, _, _ = kt.eccanom_orvara(np.array([M.to(u.rad).value]), obj.e)
        mu = obj.mu.decompose().value
        r, v = kt.orbElem2vec(E, mu, orbElem)
        x.append(r[0])
        y.append(r[1])
        z.append(r[2])
        vx.append(v[0])
        vy.append(v[1])
        vz.append(v[2])
    r = {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz, "t": times.jd}
    full_vector = pd.DataFrame(r)
    return full_vector


def prop_for_imaging_lambert(obj, t):
    # Calculates the working angle and deltaMag
    a, e, inc, w = obj.a, obj.e, obj.inc, obj.w

    M = mean_anom(obj, t)
    E = kt.eccanom(M.to(u.rad).value, obj.e)
    nu = kt.trueanom(E, e) * u.rad
    theta = nu + w

    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    s = (r / 4) * np.sqrt(
        4 * np.cos(2 * inc)
        + 4 * np.cos(2 * theta)
        - 2 * np.cos(2 * inc - 2 * theta)
        - 2 * np.cos(2 * inc + 2 * theta)
        + 12
    )
    beta = np.arccos(-np.sin(inc) * np.sin(theta))
    # For gas giants
    # p_phi = self.calc_p_phi(beta, photdict, bandinfo)
    # For terrestrial planets
    phi = lambert_func(beta)
    # p_phi = self.p * phi
    p_phi = obj.p * phi

    WA = np.arctan(s / obj.dist).decompose()
    dMag = -2.5 * np.log10(p_phi * ((obj.radius / r).decompose()) ** 2).value
    return WA, dMag


def prop_for_imaging(obj, t):
    # Calculates the working angle and deltaMag
    a, e, inc, w = obj.a, obj.e, obj.inc, obj.w
    if obj.circular:
        theta = mean_anom(obj, t)
        # theta = (math.tau * (phase - np.floor(phase))) * u.rad
        r = a
    else:
        M = mean_anom(obj, t)
        E = kt.eccanom(M.value, e)
        nu = kt.trueanom(E, e) * u.rad
        theta = nu + w
        r = a * (1 - e**2) / (1 + e * np.cos(nu))

    s = r * np.sqrt(1 - np.sin(inc) ** 2 * np.sin(theta) ** 2)
    beta = np.arccos(-np.sin(inc) * np.sin(theta))
    phi = realSolarSystemPhaseFunc(beta, obj.phiIndex)

    WA = np.arctan(s / obj.dist_to_star).decompose()
    dMag = -2.5 * np.log10(obj.p * phi * ((obj.Rp / r).decompose()) ** 2).value

    return WA, dMag


def lambert_func(beta):
    return (np.sin(beta) * u.rad + (np.pi * u.rad - beta) * np.cos(beta)) / (
        np.pi * u.rad
    )


def update(path, spec):
    """
    Function to update library with newly generated parameters
    """
    spec_path = Path(path, "spec.json")
    if spec_path.exists():
        # Load it to see if it has to be overwritten
        with open(spec_path, "r") as f:
            old_spec = json.load(f)

        # If they're the same then nothing needs to be done
        if spec == old_spec:
            needs_update = False
        else:
            needs_update = True
    else:
        needs_update = True

    if needs_update:
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        logger.info(f"Saved new specification to {spec_path}.")
    else:
        logger.info(f"Found up to date specification at {spec_path}.")


def prev_best_fit(dir, survey_name):
    """
    Function looks at a directory of orbit fits

    Args:
        dir (Path):
            The star's directory

    Returns:
        has_fit (bool):
            Whether the star has a previous fit attempted
        prev_max (int):
            What 'max_planets' was set to during the best fit attempted on
            the star
        fitting_done (bool):
            True when a search was conducted that returned less planets than
            the allowed 'max_planets', indicating that all planets that can
            be found were found.

    """
    dir_list = list(Path(dir).glob("*"))
    # Get all survey specifications
    other_specs = []
    other_specs_dirs = []
    for dir in dir_list:
        spec_file = Path(dir, "obs_spec.json")
        with open(spec_file, "r") as f:
            obs_spec = json.load(f)
        if survey_name in str(dir):
            survey_spec = obs_spec
        else:
            other_specs.append(obs_spec)
            other_specs_dirs.append(dir)

    # Conditions to use a fit from a different set of observations
    # Must have the same or fewer years of observations
    # If it has the same best precision it has to have a
    # shorter baseline for that instrument
    survey_best_precision = 100
    for inst_key in survey_spec["instruments"].keys():
        inst = survey_spec["instruments"][inst_key]
        if inst["precision"] < survey_best_precision:
            survey_best_precision = inst["precision"]
            survey_best_precision_baseline = inst["end_time"] - inst["start_time"]
    best_candidate = {
        "precision": 100,
        "baseline": 0,
        "folder": None,
        "search_path": None,
        "prev_max": 0,
        "fitting_done": False,
    }
    for other_spec, other_spec_dir in zip(other_specs, other_specs_dirs):
        condition_1 = other_spec["obs_baseline"] <= survey_spec["obs_baseline"]
        best_inst_precision = 100
        for inst_key in other_spec["instruments"].keys():
            inst = other_spec["instruments"][inst_key]
            if inst["precision"] < best_inst_precision:
                best_inst_precision = inst["precision"]
                best_inst_baseline = inst["end_time"] - inst["start_time"]
        condition_2 = (
            best_inst_precision >= survey_best_precision
            and best_inst_baseline >= survey_best_precision_baseline
        )
        (
            other_spec_has_fit,
            other_spec_prev_max,
            other_spec_fitting_done,
            other_spec_search_file,
        ) = check_orbitfit_dir(other_spec_dir, None)
        if condition_1 and condition_2 and other_spec_has_fit:
            if best_inst_precision == best_candidate["precision"]:
                # Choose candidate with better baseline in sitations where
                # they're the same
                if best_inst_precision > best_candidate["baseline"]:
                    pass
            elif best_inst_precision < best_candidate["precision"]:
                # Choose the current candidate
                best_candidate["precision"] = best_inst_precision
                best_candidate["baseline"] = best_inst_baseline
                best_candidate["folder"] = other_spec_dir
                best_candidate["search_path"] = other_spec_search_file
                best_candidate["prev_max"] = other_spec_prev_max
                best_candidate["fitting_done"] = other_spec_fitting_done

    return best_candidate


def replace_EXOSIMS_system(SS, sInd, system):
    SU = SS.SimulatedUniverse
    star_planets = np.where(SU.plan2star == sInd)[0]
    # print(
    #     (
    #         f"Adding {len(system.planets)} to {system.star.name}, sInd {sInd}."
    #         f"\n{system.getpattr('a')}"
    #     )
    # )
    if len(star_planets) > 0:
        first_ind = star_planets[0]
        last_ind = star_planets[-1]
    else:
        first_ind = np.where((SU.plan2star - sInd) > 0)[0][0]
        last_ind = first_ind - 1
    # else:
    #     breakpoint()
    #     # Don't want to add planets past the typical occurrence rate so if
    #     # there aren't any planets around a system find the closest sInd with
    #     # the same number of planets
    #     counts, bins = np.histogram(
    #         SU.plan2star, bins=np.arange(SU.plan2star[0], SU.plan2star[-1])
    #     )
    #     matching_inds = bins[np.where(counts == len(system.planets))]
    #     replacement_ind = matching_inds[np.argsort(abs(matching_inds - sInd))[0]]
    #     if len(system.planets) == 1:
    #         first_ind = replacement_ind
    #         last_ind = replacement_ind
    #     else:
    #         inds = np.where(SU.plan2star == replacement_ind)[0]
    #         first_ind = inds[0]
    #         last_ind = inds[-1]
    atts = ["a", "e", "I", "O", "w", "M0", "Mp", "Rp", "p"]
    RVtoImaging_atts = ["a", "e", "inc", "W", "w", "M0", "mass", "radius", "p"]
    for att, rvtools_att in zip(atts, RVtoImaging_atts):
        # Get the parts of the array before and after the star to be replaced
        att_arr = getattr(SU, att)
        first_part = att_arr[:first_ind]
        last_part = att_arr[last_ind + 1 :]
        if att == "p":
            setattr(
                SU,
                att,
                np.array(
                    list(
                        itertools.chain(
                            first_part,
                            SU.PlanetPopulation.get_p_from_Rp(
                                system.getpattr("radius")
                            ),
                            last_part,
                        )
                    ),
                ),
            )
        elif type(att_arr) == u.Quantity:
            unit = att_arr.unit
            setattr(
                SU,
                att,
                list(
                    itertools.chain(
                        first_part.value,
                        system.getpattr(rvtools_att).to(unit).value,
                        last_part.value,
                    )
                )
                * unit,
            )
        else:
            setattr(
                SU,
                att,
                np.array(
                    list(
                        itertools.chain(
                            first_part, system.getpattr(rvtools_att), last_part
                        )
                    ),
                ),
            )

    # Edit the plan2star
    first_part = SU.plan2star[:first_ind]
    last_part = SU.plan2star[last_ind + 1 :]
    SU.plan2star = np.array(
        list(
            itertools.chain(first_part, np.ones(len(system.planets)) * sInd, last_part)
        ),
        dtype=int,
    )
    return SS


def check_orbitfit_dir(dir, fit_dir):
    dir_list = list(Path(dir).glob("*"))
    prev_run_dirs = [folder for folder in dir_list if folder.is_dir()]
    search_exists = [Path(folder, "search.pkl").exists() for folder in prev_run_dirs]
    has_fit = False
    search_file = None
    # Must have the same or fewer max_planets
    if sum(search_exists) == 0:
        # Check to see if there was already an attempt that failed
        prev_attempt_spec = Path(fit_dir, "spec.json")
        if prev_attempt_spec.exists():
            has_fit = True
            with open(prev_attempt_spec, "r") as f:
                run_info = json.load(f)
            if not run_info["mcmc_success"]:
                prev_max = 0
                fitting_done = True
        else:
            # No attempts at orbit fitting for this system
            prev_max = 0
            fitting_done = False
    else:
        # Has a previous attempt to do orbit fitting
        prev_max = 0
        highest_planets_fitted = 0
        # search_file = Path(prev_run_dirs[0], "search.pkl")
        for prev_run in prev_run_dirs:
            prev_run_spec = Path(prev_run, "spec.json")
            if prev_run_spec.exists():
                has_fit = True
                with open(prev_run_spec, "r") as f:
                    run_info = json.load(f)

                # Get the information on that fit
                run_max = run_info["max_planets"]
                planets_fitted = run_info["planets_fitted"]

                # If more planets were searched for than previous runs
                if run_max > prev_max:
                    prev_max = run_max
                    highest_planets_fitted = planets_fitted
                    search_file = Path(prev_run, "search.pkl")

            if highest_planets_fitted < prev_max:
                fitting_done = True
            else:
                fitting_done = False

    return has_fit, prev_max, fitting_done, search_file


def count_DRM_det(DRM, status=None):
    """Build lists of detections in ensemble results."""

    nobs = len(DRM)
    ndet = 0
    plan_inds = np.array([], dtype=int)
    for obs in range(nobs):
        if "det_status" in DRM[obs].keys():
            det_status = DRM[obs]["det_status"]
            if status is None:  # unique detections
                mask = [j for j, k in enumerate([x == 1 for x in det_status]) if k]
                plan_inds = np.append(plan_inds, np.array(DRM[obs]["plan_inds"])[mask])
                ndet = np.unique(plan_inds).size
            else:
                ndet += sum([int(x == status) for x in det_status])
    return ndet


def count_DRM_char(DRM, status=None):
    """Build lists of characterizations in ensemble results.

    Args:
        drms:
            all the drms from the run_ensemble
        status (integer)
            Characterization status:
            1:full spectrum, -1:partial spectrum, 0:not characterized

    Returns:
        char (list):
            List of characterization results, depending on the characterization status

    """
    nobs = len(DRM)
    nchar = 0
    for obs in range(nobs):
        if "char_status" in DRM[obs].keys():
            # nchar += len([x for x in DRM[obs]['char_status'] if x == status])
            for planet_char in DRM[obs]["char_status"]:
                if status == "any":
                    nchar += 1
                else:
                    nchar += planet_char == status
    return nchar


def compare_schedule(builder):
    rvdata = builder.rvdata
    stars = list(
        [obs.star_name.replace("P", "P ") for obs in rvdata.scheduler.schedule]
    )
    SS = rvdata.pdet.SS
    # baseSU = rvdata.universe.SU
    # system_dump = baseSU.dump_systems()
    # SS.SimulatedUniverse.load_systems(system_dump)
    # prefilter_SS = copy.deepcopy(SS)
    # prefilter_sInds =
    # [np.argwhere(SS.TargetList.Name == star)[0][0] for star in stars]
    # pInds = np.sort(
    #     np.concatenate(
    #         [
    #             np.where(SS.SimulatedUniverse.plan2star == x)[0]
    #             for x in np.arange(SS.TargetList.nStars)
    #         ]
    #     )
    # )
    # SS.SimulatedUniverse.revise_planets_list(pInds)
    # SS.TargetList.completeness_filter_original()
    SS.run_sim()
    original_DRM = copy.deepcopy(SS.DRM)
    SS.reset_sim(genNewPlanets=False)
    # # Set up sim with the optimization problem time windows
    # SS.TargetList.completeness_filter_save(prefilter_sInds)
    # forced_sInds = np.unique(
    #     [np.argwhere(SS.TargetList.Name == star)[0][0] for star in stars]
    # )
    # breakpoint()
    # SS.SimulatedUniverse.revise_stars_list(new_sInds)
    new_sInds = [np.argwhere(SS.TargetList.Name == star)[0][0] for star in stars]
    # breakpoint()
    # pInds = np.sort(
    #     np.concatenate(
    #         [
    #             np.where(SS.SimulatedUniverse.plan2star == x)[0]
    #             for x in np.arange(SS.TargetList.nStars)
    #         ]
    #     )
    # )
    # SS.SimulatedUniverse.revise_planets_list(pInds)
    # for i, ind in enumerate(sInds):
    #     SS.SimulatedUniverse.plan2star[
    #         np.where(SS.SimulatedUniverse.plan2star == ind)[0]
    #     ] = i
    # breakpoint()
    systems = rvdata.universe.systems
    systems_to_use = []
    for sInd in new_sInds:
        system_name = SS.TargetList.Name[sInd]
        system = [
            system
            for system in systems
            if system.star.name == system_name.replace(" ", "_")
        ][0]
        # SS = replace_EXOSIMS_system(SS, sInd, system)
        systems_to_use.append(system)
    # # SS.TargetList.sInds_to_save = list(rvdata.scheduler.all_coeffs.keys())
    # SS.SimulatedUniverse.init_systems()
    # SS.SimulatedUniverse.nPlans = len(SS.SimulatedUniverse.plan2star)
    # allModes = SS.OpticalSystem.observingModes
    # num_char_modes = len(
    #     list(filter(lambda mode: "spec" in mode["inst"]["name"], allModes))
    # )
    # SS.fullSpectra = np.zeros((num_char_modes, SS.SimulatedUniverse.nPlans),
    # dtype=int)
    # SS.partialSpectra = np.zeros(
    #     (num_char_modes, SS.SimulatedUniverse.nPlans), dtype=int
    # )
    # if SS.SimulatedUniverse.earthPF:
    #     SS.SimulatedUniverse.phiIndex = (
    #         np.ones(SS.SimulatedUniverse.nPlans, dtype=int) * 2
    #     )  # Used to switch select specific phase function for each planet
    # else:
    #     SS.SimulatedUniverse.phiIndex = np.asarray([])

    SS.instantiate_forced_observations(rvdata.scheduler.schedule, systems_to_use)
    SS.run_sim()

    results = {}
    results["raw_dets"] = count_DRM_det(original_DRM)
    print(f"\n\nraw detections: {results['raw_dets']}")
    results["raw_chars"] = count_DRM_char(original_DRM)
    print(f"raw characterizations: {results['raw_chars']}")
    results["scheduled_dets"] = count_DRM_det(SS.DRM)
    print(f"scheduled detections: {results['scheduled_dets']}\n\n")

    # results["scheduled_chars"] = count_DRM_char(SS.DRM)
    # print(f"scheduled characterizations: {results['scheduled_chars']}")
    # raw_char = count_DRM_char(original_DRM)
    # percursor_char = count_DRM_char(SS.DRM)
    return results


def flatten_dict(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def overwrite_script(specs, overwrites):
    # Overwrite the specification with the overwrites
    # Overwrite the base values
    specs.update(overwrites["base"])

    # Overwrite the observing mode values
    obs_modes = overwrites.get("observingModes")
    if obs_modes:
        print("THIS IS NOT IMPLEMENTED YET")
        print("Probably need to key on both instName and systName")
        breakpoint()

    # Overwrite the science instruments
    insts = overwrites.get("scienceInstruments")
    if insts:
        for key, val in insts.items():
            for _dict in specs["scienceInstruments"]:
                if _dict["name"] == key:
                    _dict.update(val)

    # Overwrite starlight suppression systems
    sss = overwrites.get("starlightSuppressionSystems")
    if sss:
        for key, val in sss.items():
            for _dict in specs["starlightSuppressionSystems"]:
                if _dict["name"] == key:
                    _dict.update(val)
    # Propagate the changes to the loaded script
    return specs


def EXOSIMS_script_hash(script_path, specs=None):
    valid_types = [str, int, float, u.quantity.Quantity, type(None)]
    if specs is None:
        with open(script_path) as f:
            specs = json.loads(f.read())
    flattened_dict = flatten_dict(specs)
    all_info = []
    rejected = []
    list_of_dicts = [
        "scienceInstruments",
        "starlightSuppressionSystems",
        "observingModes",
    ]
    for key in list_of_dicts:
        key_list = flattened_dict[key]
        for _dict in key_list:
            if "name" in _dict.keys():
                _name = _dict["name"]
            elif "instName" in _dict.keys():
                _name = f"{_dict['instName']}{_dict['systName']}"
            for sub_key, sub_val in _dict.items():
                if (sub_key not in ["name", "instName", "systName"]) and (
                    type(sub_val) in valid_types
                ):
                    all_info.append(f"{key}{_name}{sub_key}{sub_val}")
                    flattened_dict[f"{key}{_name}{sub_key}"] = sub_val
                else:
                    rejected.append(_dict[sub_key])
        flattened_dict.pop(key)
    hash = genHexStr(dictToSortedStr(specs))
    return hash
