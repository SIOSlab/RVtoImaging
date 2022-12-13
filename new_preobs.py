import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time

from RVtools.builder import BaseBuilder, Director

if __name__ == "__main__":
    # Load settings for this machine
    settings_file = Path(".config.json")
    with open(settings_file, "r") as f:
        settings = json.load(f)
    cache_dir = settings["cache_dir"]
    workers = settings["workers"]
    first_seed = settings["first_seed"]
    last_seed = settings["last_seed"]

    # Set up director and builder objects
    director = Director()
    builder = BaseBuilder(cache_dir=cache_dir, workers=workers)
    director.builder = builder

    ######################################################################
    # Set up universe generation
    ######################################################################
    builder.universe_params = {
        "universe_type": "exosims",
        "script": "test.json",
    }

    ######################################################################
    # Set up precursor observation information
    ######################################################################
    # Create base instrument parameters
    mission_start = Time(2043, format="decimalyear")

    # Create instrument bases
    rv100_25 = {
        "name": "1 m/s",
        "precision": 1 * u.m / u.s,
        "start_time": mission_start - 20 * u.yr,
    }

    rv40_15 = {
        "name": "40 cm/s",
        "precision": 0.4 * u.m / u.s,
        "start_time": mission_start - 15 * u.yr,
    }

    rv10_15 = {
        "name": "10 cm/s",
        "precision": 0.1 * u.m / u.s,
        "start_time": mission_start - 15 * u.yr,
    }

    rv03_15 = {
        "name": "3 cm/s",
        "precision": 0.03 * u.m / u.s,
        "start_time": mission_start - 15 * u.yr,
    }

    survey2 = {
        "fit_order": 1,
        "instruments": [rv100_25, rv40_15, rv10_15],
    }
    survey3 = {
        "fit_order": 2,
        "instruments": [rv100_25, rv40_15, rv03_15],
    }
    surveys = [survey3]

    # Save parameters to the builder
    base_params = {
        "observation_scheme": "survey",
        "observations_per_night": 20,
        "bad_weather_prob": 0.5,
        "end_time": mission_start,
    }
    nsystems = 125
    systems = np.arange(nsystems)
    builder.preobs_params = {
        "base_params": base_params,
        "surveys": surveys,
        "n_systems_to_observe": nsystems,
        "filters": ["distance"],
    }

    ######################################################################
    # Orbit fitting
    ######################################################################
    builder.orbitfit_params = {
        "fitting_method": "rvsearch",
        "max_planets": 3,
    }

    # RUN THE SEEDS
    seeds = [int(seed) for seed in np.arange(first_seed, last_seed, 1)]
    director.run_seeds(seeds)
    ######################################################################
    # Probability of detection
    ######################################################################
    # builder.pdet_params = {
    #     "construction_method": "multivariate gaussian",
    #     "cov_samples": 1000,
    #     "number_of_orbits": 1000,
    #     "systems_of_interest": [0],
    #     "start_time": mission_start,
    #     "end_time": mission_start + 10 * u.yr,
    # }

    # builder.probability_of_detection()

    # builder.precursor_data.list_parts()

    # print("\n")

    # print("Standard full featured precursor_data: ")
    # director.build_full_info()
    # builder.precursor_data.list_parts()

    # print("\n")

    # # Remember, the Builder pattern can be used without a Director class.
    # print("Custom precursor_data: ")
    # builder.create_universe()
    # builder.simulate_observations()
    # builder.precursor_data.list_parts()
