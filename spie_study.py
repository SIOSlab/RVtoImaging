import json
from itertools import product
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astroplan import AirmassConstraint, AtNightConstraint
from astropy.time import Time

from RVtoImaging.builder import BaseBuilder

if __name__ == "__main__":
    # Load settings for this machine
    settings_file = Path(".config.json")
    with open(settings_file, "r") as f:
        settings = json.load(f)
    cache_dir = settings["cache_dir"]
    workers = settings["workers"]

    # Set up builder object
    builder = BaseBuilder(cache_dir=cache_dir, workers=workers)

    target_list = ".cache/noise_109.csv"

    ######################################################################
    # Set up universe generation
    ######################################################################
    builder.universe_params = {
        "universe_type": "exovista",
        "target_list": target_list,
        "convert": True,
        "filter": True,
    }
    # builder.build_universe()
    # universe = builder.rv2img.universe
    # sigma_mag = u.Quantity([system.star.sigma_mag for system in universe.systems])
    # sigma_gran = u.Quantity([system.star.sigma_gran for system in universe.systems])
    # system_names = [system.star.name for system in universe.systems]
    # # Create pandas DataFrame for those values
    # df = pd.DataFrame(
    #     {
    #         "system": system_names,
    #         "sigma_mag": sigma_mag,
    #         "sigma_gran": sigma_gran,
    #     }
    # )
    # df["sigma_quadrature"] = np.sqrt(df["sigma_mag"] ** 2 + df["sigma_gran"] ** 2)
    # breakpoint()

    ######################################################################
    # Orbit fitting
    ######################################################################
    builder.rv_fits_params = {
        "fitting_method": "rvsearch",
        "max_planets": 5,
        "vary_planets": np.inf,
    }

    ######################################################################
    # Set up precursor observation information
    ######################################################################
    # Create base instrument parameters
    mission_start = Time(2043, format="decimalyear")

    # EPRV observation run
    EPRV_observation_scheme = {
        "type": "constraint",
        "bad_weather_prob": 0,
        "obs_night_schedule": "random",
        "n_obs_nights": 100,
        "exposure_time": 20 * u.min,
        "log_search_progress": True,
        "requested_observations": 100,
        "max_time_in_seconds": 10 * 60,
        "astroplan_constraints": [
            AtNightConstraint.twilight_astronomical(),
            AirmassConstraint(min=1, max=1.5),
        ],
    }

    ######################################################################
    # Probability of detection
    ######################################################################
    builder.pdet_params = {
        "script": "scripts/spie.json",
        "construction_method": {"name": "multivariate gaussian", "cov_samples": 1000},
        "number_of_orbits": 10000,
        "start_time": mission_start,
        "end_time": mission_start + 2 * u.yr,
        "min_int_time": 1 * u.hr,
        "max_int_time": 30 * u.day,
        "fEZ_quantile": 0.75,
        "fEZ_exact": True,
        "EXOSIMS_overwrites": {
            "base": {"catalogpath": target_list},
            "starlightSuppressionSystems": {"VVC500": {"koAngles_Sun": [75, 105]}},
        },
        "include_keepout_in_pdet": False,
    }
    builder.img_schedule_params = {
        "coeff_multiple": 100,
        "sim_length": 2 * u.yr,
        "block_length": 2 * u.hr,
        "block_multiples": [3, 12, 5 * 12, 10 * 12, 30 * 12],
        "max_observations_per_star": 10,
        "planet_threshold": 1,
        "requested_planet_observations": 3,
        "min_required_wait_time": 10 * u.d,
        "max_required_wait_time": 0.25 * u.yr,
        "max_time_in_seconds": 20 * 60,
        "opt": {
            "log_search_progress": True,
            "random_walk_method": "int_times",
            "n_random_walks": 1,
        },
    }
    precision = 0.01 * u.m / u.s
    EPRV_run = {
        "name": f"EPRV_{precision.value:.3f}",
        "location": "W. M. Keck Observatory",
        "timezone": "US/Hawaii",
        "start_time": Time(2033, format="decimalyear"),
        "end_time": Time(2043, format="decimalyear"),
        "observation_scheme": EPRV_observation_scheme,
        "sigma_terms": {"sigma_inst": precision},
        "rv_noise_mitigation": {
            "sigma_mag": 5 * u.m / u.s,
            "sigma_gran": 5 * u.m / u.s,
        },
        "rv_noise_cutoff": 0.2 * u.m / u.s,
    }
    builder.rv_dataset_params = {
        "dataset_name": f"EPRV_{precision.value:.3f}",
        "rv_observing_runs": [EPRV_run],
        "available_targets_file": target_list,
        "approx_systems_to_observe": 100,
    }

    ram_angles_toward = np.arange(5, 46, 5)
    ram_angles_away = np.arange(5, 46, 5)
    seeds = np.arange(0, 4, 1).tolist()
    runs_file = Path("FOR_study_runs.csv")
    if not runs_file.exists():
        columns = [
            "seed",
            "ram_angle_toward",
            "ram_angle_away",
            "universe",
            "result_dir",
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(runs_file, index=False)
    else:
        df = pd.read_csv(runs_file)

    for seed in seeds:
        for ram_angle_toward, ram_angle_away in product(
            ram_angles_toward, ram_angles_away
        ):
            # Check if the seed and ram angles have already been run
            if (
                df[
                    (df["seed"] == seed)
                    & (df["ram_angle_toward"] == ram_angle_toward)
                    & (df["ram_angle_away"] == ram_angle_away)
                ].shape[0]
                > 0
            ):
                continue
            koAngles_Sun = [90 - int(ram_angle_toward), 90 + int(ram_angle_away)]
            _dict = {"VVC500": {"koAngles_Sun": koAngles_Sun}}
            builder.pdet_params["EXOSIMS_overwrites"][
                "starlightSuppressionSystems"
            ] = _dict
            builder.run_sim(seed)
            universe_hash = builder.rv2img.universe_hash

            universe_location = Path(
                ".cache", f"universe_{builder.rv2img.universe_hash}"
            )
            results_dir = Path(builder.rv2img.scheduler.result_path)
            # Append new row to DataFrame
            new_row = pd.DataFrame(
                [
                    {
                        "seed": seed,
                        "ram_angle_toward": ram_angle_toward,
                        "ram_angle_away": ram_angle_away,
                        "universe": str(universe_location),
                        "result_dir": str(results_dir),
                    }
                ]
            )
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(runs_file, index=False)
