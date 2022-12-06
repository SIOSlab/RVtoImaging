import astropy.units as u
import numpy as np
from astropy.time import Time

from RVtools.builder import BaseBuilder, Director

if __name__ == "__main__":
    director = Director()
    builder = BaseBuilder()
    director.builder = builder

    # Caching
    # hash = f"{random.getrandbits(128):032x}"[:8]
    builder.run_title = "test"

    ######################################################################
    # Set up universe generation
    ######################################################################
    # builder.universe_type = "exovista"
    # builder.universe_params = {"data_path": "./data/", "universe_number": 1}
    # builder.universe_type = "exosims"
    nsystems = 500
    builder.cache_universe = True
    builder.universe_params = {
        "universe_type": "exosims",
        "script": "test.json",
        "nsystems": nsystems,
    }
    director.build_universe()
    print(builder.rvdata.universe.cache_path)

    ######################################################################
    # Set up precursor observation information
    ######################################################################
    # Create base instrument parameters
    base_params = {
        "timing_format": "Poisson",
        # "observation_scheme": "time_cluster",
        "observation_scheme": "survey",
        "targets_per_observation": 5,
        "cluster_length": 30 * u.d,
        "cluster_choice": "random",
    }
    # Create instrument specific parameters
    # eprv = {
    #     "name": "EPRV",
    #     "precision": 0.05 * u.m / u.s,
    #     "rate": 0.25 / u.d,
    #     "start_time": Time(5, format="decimalyear"),
    #     "end_time": Time(10, format="decimalyear"),
    # }
    # prv = {
    #     "name": "PRV",
    #     "precision": 0.4 * u.m / u.s,
    #     "rate": 0.5 / u.d,
    #     "start_time": Time(2.5, format="decimalyear"),
    #     "end_time": Time(10, format="decimalyear"),
    # }

    # rv = {
    #     "name": "RV",
    #     "precision": 1.5 * u.m / u.s,
    #     "rate": 1 / u.d,
    #     "start_time": Time(0, format="decimalyear"),
    #     "end_time": Time(5, format="decimalyear"),
    # }

    # Create instrument specific parameters
    eprv = {
        "name": "EPRV",
        "precision": 0.02 * u.m / u.s,
        "rate": 5 / u.d,
        "start_time": Time(15, format="decimalyear"),
        "end_time": Time(21, format="decimalyear"),
    }
    prv = {
        "name": "PRV",
        "precision": 0.4 * u.m / u.s,
        "rate": 5 / u.d,
        "start_time": Time(5, format="decimalyear"),
        "end_time": Time(21, format="decimalyear"),
    }

    rv = {
        "name": "RV",
        "precision": 1.5 * u.m / u.s,
        "rate": 5 / u.d,
        "start_time": Time(0, format="decimalyear"),
        "end_time": Time(15, format="decimalyear"),
    }

    # Save parameters to the builder
    systems = np.arange(nsystems)
    builder.preobs_params = {
        "base_params": base_params,
        "instruments": [eprv, prv, rv],
        "systems_to_observe": systems.tolist(),
    }
    builder.cache_preobs = True
    builder.simulate_rv_observations()

    ######################################################################
    # Orbit fitting
    ######################################################################
    builder.orbitfit_params = {
        "fitting_method": "rvsearch",
        "max_planets": 1,
        "systems_to_fit": [0, 1, 3, 5, 8, 10, 15],
        "dynamic_max": True,
        # "mcmc_timeout_mins": 1.5,
    }
    builder.cache_orbitfit = True
    builder.orbit_fitting()

    ######################################################################
    # Probability of detection
    ######################################################################
    builder.construction_params = {"construction_method": "credible interval"}

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
