import json
import subprocess
from pathlib import Path

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import pandas as pd
from keplertools import fun as kt
from tqdm import tqdm


def runcmd(cmd, verbose=False):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def get_data(universes=np.arange(1, 13)):
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
        Path(f"data/{n}/").mkdir(parents=True, exist_ok=True)
        if not Path(f"data/{n}/target_database.csv").exists():
            runcmd(f"wget --directory-prefix=data/{n} {universe_url}", verbose=False)
        df = pd.read_csv(f"data/{n}/target_database.csv", low_memory=False)
        for i in tqdm(
            np.arange(1, df.shape[0]), position=1, desc="System", leave=False
        ):
            # Get file url
            fit_url = df.at[i, "URL"]

            # Create file path
            file_path = Path(f"data/{n}/{fit_url.split('/')[-1]}")

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


def calc_position_vectors(obj, times):
    orbElem = (
        obj.a.decompose().value,
        obj.e,
        obj.W.to(u.rad).value,
        obj.inc.to(u.rad).value,
        obj.w.to(u.rad).value,
    )
    x, y, z, vx, vy, vz = [], [], [], [], [], []
    for t in times:
        M = mean_anom(obj, t)
        E = kt.eccanom(M.to(u.rad).value, obj.e)
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
        ) = check_orbitfit_dir(other_spec_dir)
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


def check_orbitfit_dir(dir):
    dir_list = list(Path(dir).glob("*"))
    prev_run_dirs = [folder for folder in dir_list if folder.is_dir()]
    search_exists = [Path(folder, "search.pkl").exists() for folder in prev_run_dirs]
    has_fit = False
    search_file = None
    # Must have the same or fewer max_planets
    if sum(search_exists) == 0:
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
