import astropy.units as u
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
    builder.cache_universe = True
    builder.universe_params = {
        "universe_type": "exosims",
        "script": "test.json",
        "nsystems": 20,
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
        "rate": 0.5 / u.d,
        "start_time": Time(7.5, format="decimalyear"),
        "end_time": Time(10, format="decimalyear"),
    }
    prv = {
        "name": "PRV",
        "precision": 0.4 * u.m / u.s,
        "rate": 0.5 / u.d,
        "start_time": Time(2.5, format="decimalyear"),
        "end_time": Time(7.5, format="decimalyear"),
    }

    rv = {
        "name": "RV",
        "precision": 1.5 * u.m / u.s,
        "rate": 0.33 / u.d,
        "start_time": Time(0, format="decimalyear"),
        "end_time": Time(5, format="decimalyear"),
    }

    # Save parameters to the builder
    builder.preobs_params = {
        "base_params": base_params,
        "instruments": [eprv, prv, rv],
        "systems_to_observe": [0, 1, 3, 5, 8, 10, 15],
    }
    builder.cache_preobs = True
    builder.simulate_rv_observations()

    ######################################################################
    # Orbit fitting
    ######################################################################
    builder.orbit_fitting_params = {
        "fitting_method": "rvsearch",
        "max_planets": 2,
        "systems_to_fit": [8, 10, 15],
        "dynamic_max": True,
        "mcmc_timeout_mins": 0.01,
    }
    builder.orbit_fitting()

    ######################################################################
    # Orbit construction
    ######################################################################
    builder.construction

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
