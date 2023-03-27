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
    first_seed = settings["first_seed"]
    last_seed = settings["last_seed"]

    # Set up director and builder objects
    # director = Director()
    builder = BaseBuilder(cache_dir=cache_dir, workers=workers)
    # director.builder = builder

    ######################################################################
    # Set up universe generation
    ######################################################################
    builder.universe_params = {
        "universe_type": "exosims",
        "script": "test.json",
    }
    # builder.universe_params = {
    #     "universe_type": "exosims",
    #     "data_path": "data/",
    #     "universe_number": 1,
    # }

    ######################################################################
    # Set up precursor observation information
    ######################################################################
    # Create base instrument parameters
    mission_start = Time(2043, format="decimalyear")
    # This is the location of the APF
    # APF = {
    #     "name": "APF",
    #     "location": "Lick Observatory",
    #     "precision": 1 * u.m / u.s,
    #     "start_time": mission_start - 15 * u.yr,
    #     # "end_time": mission_start * u.yr,
    #     "observation_scheme": "consistent",
    #     "observation_rate": 2 / u.week,
    # }

    # observations at 2 m/s by various instruments
    simple_sigma_terms = {"rv": 2 * u.m / u.s}
    simple_observation_scheme = {
        "type": "random",
        "observations_per_star_per_year": 15,
        "bad_weather_prob": 0.0,
        "exposure_time": 30 * u.min,
        "astroplan_constraints": [AtNightConstraint.twilight_astronomical()],
    }
    # Kinda simulating HIRES which is at Keck
    simple_run = {
        "name": "2 m/s instruments",
        "location": "W. M. Keck Observatory",
        "timezone": "US/Hawaii",
        "start_time": Time(2028, format="decimalyear"),
        "end_time": mission_start,
        "sigma_terms": simple_sigma_terms,
        "observation_scheme": simple_observation_scheme,
    }

    # NEID Earth Twin Survey values
    NETS_sigma_terms = {
        "instrument": 0.27 * u.m / u.s,
        "photon": 0.21213 * u.m / u.s,
        "pmode_oscillation": 0.21213 * u.m / u.s,
        "granulation": 1.0 * u.m / u.s,
        "magnetic": 1.0 * u.m / u.s,
    }
    NETS_observation_scheme = {
        "type": "constraint",
        "bad_weather_prob": 0,
        "obs_night_schedule": "random",
        "n_obs_nights": 30,
        "min_observations_per_star_per_year": 20,
        # "slew_overhead": 180 * u.s,
        "exposure_time": 20 * u.min,
        "astroplan_constraints": [
            AtNightConstraint.twilight_astronomical(),
            AirmassConstraint(min=1, max=1.5),
        ],
    }
    NETS_run = {
        "name": "NETS",
        "location": "Kitt Peak",
        "timezone": "US/Mountain",
        "start_time": Time(2038, format="decimalyear"),
        "end_time": mission_start,
        "observation_scheme": NETS_observation_scheme,
        "sigma_terms": NETS_sigma_terms,
    }
    observing_runs = [simple_run]
    # observing_runs = [simple_run, NETS_run]
    approx_systems_to_observe = 35
    target_list_constraints = {"": ""}
    # builder.rv_dataset_params = {
    #     "dataset_name": "simple",
    #     "rv_observing_runs": [simple_run],
    #     "available_targets_file": ".cache/NETS100.csv",
    #     "approx_systems_to_observe": approx_systems_to_observe,
    # }
    builder.rv_dataset_params = {
        "dataset_name": "NETS",
        "rv_observing_runs": [simple_run, NETS_run],
        "available_targets_file": ".cache/NETS100.csv",
        "approx_systems_to_observe": approx_systems_to_observe,
    }

    ######################################################################
    # Orbit fitting
    ######################################################################
    builder.rv_fits_params = {
        "fitting_method": "rvsearch",
        "max_planets": 4,
        "vary_planets": np.inf,
    }

    # RUN THE SEEDS
    seeds = [int(seed) for seed in np.arange(first_seed, last_seed + 1, 1)]
    builder.seeds = seeds
    builder.run_seeds()
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
