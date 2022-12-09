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
        # "nsystems": nsystems,
    }
    director.build_universe()

    ######################################################################
    # Set up precursor observation information
    ######################################################################
    # Create base instrument parameters
    base_params = {
        "timing_format": "Poisson",
        "observation_scheme": "survey",
        "targets_per_observation": 5,
        "cluster_length": 30 * u.d,
        "cluster_choice": "random",
    }

    # Create instrument specific parameters
    eprv = {
        "name": "3 cm/s",
        "precision": 0.03 * u.m / u.s,
        "rate": 2 / u.d,
        "start_time": Time(2032, format="decimalyear"),
        "end_time": Time(2043, format="decimalyear"),
    }
    prv = {
        "name": "40 cm/s",
        "precision": 0.4 * u.m / u.s,
        "rate": 2 / u.d,
        "start_time": Time(2022, format="decimalyear"),
        "end_time": Time(2043, format="decimalyear"),
    }

    rv = {
        "name": "1 m/s",
        "precision": 1 * u.m / u.s,
        "rate": 4 / u.d,
        "start_time": Time(2015, format="decimalyear"),
        "end_time": Time(2043, format="decimalyear"),
    }

    # Save parameters to the builder
    nsystems = 1000
    systems = np.arange(nsystems)
    builder.preobs_params = {
        "base_params": base_params,
        "instruments": [eprv, prv, rv],
        "systems_to_observe": systems.tolist(),
    }
    builder.simulate_rv_observations()

    ######################################################################
    # Orbit fitting
    ######################################################################
    builder.orbitfit_params = {
        "fitting_method": "rvsearch",
        "max_planets": 2,
        "systems_to_fit": np.arange(101, 120, 1).tolist(),
    }
    builder.orbit_fitting()

    ######################################################################
    # Probability of detection
    ######################################################################
    builder.pdet_params = {
        "construction_method": "multivariate gaussian",
        "number_of_orbits": 1000,
        "systems_of_interest": [0],
        "start_time": Time(20, format="decimalyear"),
        "end_time": Time(40, format="decimalyear"),
        "cov_samples": 1000,
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
