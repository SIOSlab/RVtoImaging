import json
from pathlib import Path

import astropy.units as u
import numpy as np
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
    # first_seed = settings["first_seed0"]
    # last_seed = settings["last_seed0"]

    # Set up builder object
    builder = BaseBuilder(cache_dir=cache_dir, workers=workers)

    ######################################################################
    # Set up universe generation
    ######################################################################
    builder.universe_params = {
        "universe_type": "exovista",
        "data_path": "data",
        "universe_number": 1,
    }

    ######################################################################
    # Orbit fitting
    ######################################################################
    builder.rv_fits_params = {
        "fitting_method": "rvsearch",
        "max_planets": 4,
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
    # builder.pdet_params = {
    #     "script": "scripts/caseA_manyearths.json",
    #     "construction_method": {"name": "credible interval"},
    #     "number_of_orbits": 10000,
    #     "start_time": mission_start,
    #     "end_time": mission_start + 2 * u.yr,
    #     "min_int_time": 1 * u.hr,
    #     "max_int_time": 30 * u.day,
    #     "fEZ_quantile": 0.25,
    # }
    # builder.img_schedule_params = {
    #     "coeff_multiple": 100,
    #     "sim_length": 2 * u.yr,
    #     "block_length": 2 * u.hr,
    #     "block_multiples": [3, 12, 5 * 12, 10 * 12, 30 * 12],
    #     "max_observations_per_star": 10,
    #     "planet_threshold": 0.95,
    #     "requested_planet_observations": 3,
    #     "min_required_wait_time": 10 * u.d,
    #     "max_required_wait_time": 0.25 * u.yr,
    #     "max_time_in_seconds": 1 * 60 * 60,
    #     "opt": {
    #         "log_search_progress": True,
    #         "random_walk_method": "int_times",
    #         "n_random_walks": 300,
    #     },
    # }
    precision = 0.03 * u.m / u.s
    EPRV_run = {
        "name": f"EPRV_{precision.value:.3f}",
        "location": "W. M. Keck Observatory",
        "timezone": "US/Hawaii",
        "start_time": Time(2038, format="decimalyear"),
        "end_time": Time(2043, format="decimalyear"),
        "observation_scheme": EPRV_observation_scheme,
        "sigma_terms": {"sigma_rv": precision},
    }
    builder.rv_dataset_params = {
        "dataset_name": f"EPRV_{precision.value:.3f}",
        "rv_observing_runs": [EPRV_run],
        "available_targets_file": f"{cache_dir}/temp.csv",
        "approx_systems_to_observe": 100,
    }

    builder.run_sim(1)
    # seeds = np.arange(first_seed, last_seed, 1)
    # for seed in seeds:
    #     builder.run_sim(seed)
