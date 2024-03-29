import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astroplan import AirmassConstraint, AtNightConstraint
from astropy.time import Time

from RVtoImaging.builder import BaseBuilder

# def run_scenario(seed, builder, rv_dataset_info):
# def run_scenario(args):
#     seed, builder, run_sets = args

#     builder = copy.deepcopy(builder)
#     builder.seeds = [seed]
#     for dataset_name, obs_runs in run_sets.items():
#         print(f"Seed {seed} for {dataset_name} dataset")
#         builder.rv_dataset_params = {
#             "dataset_name": dataset_name,
#             "rv_observing_runs": obs_runs,
#             "available_targets_file": ".cache/NETS100.csv",
#             "approx_systems_to_observe": approx_systems_to_observe,
#         }
#         builder.run_seeds()
#     # builder.rv_dataset_params = {
#     #     "dataset_name": dataset_name,
#     #     "rv_observing_runs": obs_runs,
#     #     "available_targets_file": ".cache/NETS100.csv",
#     #     "approx_systems_to_observe": 85,
#     # }
#     # builder.run_seeds()


# def run_search(path):
#     base_path = Path(".cache/universe_309864d7/HIP_101997/")
#     rv_df = pd.read_csv(Path(base_path, path, "rv.csv"))
#     print("test")
#     searcher = search.Search(
#         rv_df, workers=7, mcmc=True, verbose=True, max_planets=4, n_vary=np.inf
#     )
#     searcher.run_search()


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
    #     "universe_type": "exovista",
    #     "data_path": ".cache/data/",
    #     "universe_number": 0,
    # }

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
        "observations_per_star_per_year": 10,
        "bad_weather_prob": 0.0,
        "exposure_time": 30 * u.min,
        "astroplan_constraints": [
            AtNightConstraint.twilight_astronomical(),
            AirmassConstraint(min=1, max=1.5),
        ],
    }
    # Kinda simulating HIRES which at Keck
    simple_run = {
        "name": "2 m/s instruments",
        "location": "W. M. Keck Observatory",
        "timezone": "US/Hawaii",
        "start_time": Time(2010, format="decimalyear"),
        "end_time": Time(2035, format="decimalyear"),
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
        "n_obs_nights": 24,
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
        "start_time": Time(2023, format="decimalyear"),
        "end_time": Time(2028, format="decimalyear"),
        "observation_scheme": NETS_observation_scheme,
        "sigma_terms": NETS_sigma_terms,
    }

    # UPGRADE TO NEID
    NEID_upgrade_run_1 = {
        "name": "NEID 50cm/s",
        "location": "Kitt Peak",
        "timezone": "US/Mountain",
        "start_time": Time(2030, format="decimalyear"),
        "end_time": Time(2035, format="decimalyear"),
        "observation_scheme": NETS_observation_scheme,
        "sigma_terms": {"sigma_rv": 0.5 * u.m / u.s},
    }

    NEID_upgrade_run_2 = {
        "name": "NEID 40cm/s",
        "location": "Kitt Peak",
        "timezone": "US/Mountain",
        "start_time": Time(2038, format="decimalyear"),
        "end_time": Time(2043, format="decimalyear"),
        "observation_scheme": NETS_observation_scheme,
        "sigma_terms": {"sigma_rv": 0.4 * u.m / u.s},
    }

    # EPRV observation run
    EPRV_observation_scheme = {
        "type": "constraint",
        "bad_weather_prob": 0,
        "obs_night_schedule": "random",
        "n_obs_nights": 24,
        "exposure_time": 20 * u.min,
        "astroplan_constraints": [
            AtNightConstraint.twilight_astronomical(),
            AirmassConstraint(min=1, max=1.5),
        ],
    }

    EPRV_run = {
        "name": "EPRV",
        "location": "W. M. Keck Observatory",
        "timezone": "US/Hawaii",
        "start_time": Time(2038, format="decimalyear"),
        "end_time": Time(2043, format="decimalyear"),
        "observation_scheme": EPRV_observation_scheme,
        "sigma_terms": {"sigma_rv": 0.1 * u.m / u.s},
    }
    observing_runs = [simple_run]
    # observing_runs = [simple_run, NETS_run]
    approx_systems_to_observe = 84
    target_list_constraints = {"": ""}
    # builder.rv_dataset_params = {
    #     "dataset_name": "simple",
    #     "rv_observing_runs": [simple_run],
    #     "available_targets_file": ".cache/NETS100.csv",
    #     "approx_systems_to_observe": approx_systems_to_observe,
    # }
    baseline_sigma_terms = {"rv": 1 * u.m / u.s}
    baseline_observation_scheme = {
        "type": "random",
        "observations_per_star_per_year": 10,
        "bad_weather_prob": 0.0,
        "exposure_time": 30 * u.min,
        "astroplan_constraints": [
            AtNightConstraint.twilight_astronomical(),
            AirmassConstraint(min=1, max=1.5),
        ],
    }
    # Kinda simulating HIRES which at Keck
    baseline_run = {
        "name": "1 m/s instruments",
        "location": "W. M. Keck Observatory",
        "timezone": "US/Hawaii",
        "start_time": Time(2010, format="decimalyear"),
        "end_time": Time(2043, format="decimalyear"),
        "sigma_terms": baseline_sigma_terms,
        "observation_scheme": baseline_observation_scheme,
    }
    baseline = [baseline_run]
    conservative = [simple_run, NETS_run, NEID_upgrade_run_1, NEID_upgrade_run_2]
    optimistic = [simple_run, NETS_run, NEID_upgrade_run_1, EPRV_run]
    run_sets = {
        "conservative": conservative,
        "baseline": baseline,
        "optimistic": optimistic,
    }

    seeds = [int(seed) for seed in np.arange(first_seed, last_seed + 1, 1)]
    builder.rv_dataset_params = {
        "dataset_name": "conservative",
        "rv_observing_runs": conservative,
        "available_targets_file": f"{cache_dir}/NETS100.csv",
        "approx_systems_to_observe": approx_systems_to_observe,
    }

    # for seed in seeds:
    #     builder.seeds = [seed]
    #     for dataset_name, obs_runs in run_sets.items():
    #         builder.rv_dataset_params = {
    #             "dataset_name": dataset_name,
    #             "rv_observing_runs": obs_runs,
    #             "available_targets_file": f"{cache_dir}/NETS100.csv",
    #             "approx_systems_to_observe": approx_systems_to_observe,
    #         }
    #         builder.run_seeds()
    # args = []
    # for seed in seeds:
    #     args.append((seed, builder, run_sets))

    ######################################################################
    # Probability of detection
    ######################################################################
    builder.pdet_params = {
        "script": "scripts/caseA.json",
        "construction_method": {"name": "multivariate gaussian", "cov_samples": 1000},
        "number_of_orbits": 10000,
        "start_time": mission_start,
        "end_time": mission_start + 2 * u.yr,
        "min_int_time": 1 * u.hr,
        "max_int_time": 10 * u.day,
    }

    builder.img_schedule_params = {
        "coeff_multiple": 10,
        "sim_length": 1 * u.yr,
        "window_length": 2 * u.hr,
        "block_multiples": [1, 6, 36],
        "max_observations_per_star": 10,
        "planet_threshold": 0.9,
        "requested_planet_observations": 3,
        "min_required_wait_time": 10 * u.d,
        "max_required_wait_time": 0.25 * u.yr,
        "max_time_in_seconds": 60 * 60,
        "log_search_progress": True,
    }
    builder.seeds = seeds
    builder.run_seeds()

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
