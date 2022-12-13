import astropy.units as u
import numpy as np
from astropy.time import Time

from RVtools.builder import BaseBuilder, Director

if __name__ == "__main__":
    director = Director()
    builder = BaseBuilder(cache_dir=".cache", workers=14)
    director.builder = builder

    ######################################################################
    # Set up universe generation
    ######################################################################
    # builder.universe_type = "exovista"
    # builder.universe_params = {"data_path": "./data/", "universe_number": 1}
    builder.universe_params = {
        "universe_type": "exosims",
        "script": "test.json",
    }
    director.build_universe()

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

    # Create surveys
    # survey1 = {
    #     "name": "100cm_25yr",
    #     "priority": 0,
    #     "instruments": [rv100_25],
    # }
    survey2 = {
        "fit_order": 1,
        "instruments": [rv100_25, rv40_15, rv10_15],
    }
    survey3 = {
        "fit_order": 2,
        "instruments": [rv100_25, rv40_15, rv03_15],
    }
    # survey4 = {
    #     "name": "100cm_25yr-03cm_15yr",
    #     "priority": 3,
    #     "instruments": [rv100_25, rv03_15],
    # }
    # survey5 = {
    #     "name": "100cm_25yr-10cm_15yr",
    #     "priority": 4,
    #     "instruments": [rv100_25, rv10_15],
    # }
    # surveys = [survey2, survey3]
    surveys = [survey2, survey3]
    # Save parameters to the builder
    base_params = {
        # "timing_format": "Poisson",
        "observation_scheme": "survey",
        "observations_per_night": 20,
        "bad_weather_prob": 0.3,
        "end_time": mission_start,
    }
    nsystems = 150
    systems = np.arange(nsystems)
    builder.preobs_params = {
        "base_params": base_params,
        "surveys": surveys,
        "n_systems_to_observe": nsystems,
        "filters": ["distance"],
    }
    builder.simulate_rv_observations()

    ######################################################################
    # Orbit fitting
    ######################################################################
    builder.orbitfit_params = {
        "fitting_method": "rvsearch",
        "max_planets": 5,
    }
    builder.orbit_fitting()

    ######################################################################
    # Probability of detection
    ######################################################################
    builder.pdet_params = {
        "construction_method": "multivariate gaussian",
        "cov_samples": 1000,
        "number_of_orbits": 1000,
        "systems_of_interest": [0],
        "start_time": mission_start,
        "end_time": mission_start + 10 * u.yr,
    }

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
