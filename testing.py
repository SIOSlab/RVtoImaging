import astropy.units as u
from astropy.time import Time

from RVtools.builder import BaseBuilder, Director

if __name__ == "__main__":
    director = Director()
    builder = BaseBuilder()
    director.builder = builder

    ######################################################################
    # Set up universe generation
    ######################################################################
    # builder.universe_type = "exovista"
    # builder.universe_params = {"data_path": "./data/", "universe_number": 1}
    builder.universe_type = "exosims"
    builder.universe_params = {"script": "test.json", "nsystems": 20}
    director.build_universe()

    ######################################################################
    # Set up precursor observation information
    ######################################################################
    # Create base instrument parameters
    base_params = {
        "timing_format": "Poisson",
        "observation_scheme": "time_cluster",
        "cluster_length": 30 * u.d,
        "cluster_choice": "random",
    }
    # Create instrument specific parameters
    eprv = {
        "name": "EPRV",
        "precision": 0.05 * u.m / u.s,
        "rate": 0.25 / u.d,
        "start_time": Time(5, format="decimalyear"),
        "end_time": Time(10, format="decimalyear"),
    }
    prv = {
        "name": "PRV",
        "precision": 0.4 * u.m / u.s,
        "rate": 0.5 / u.d,
        "start_time": Time(2.5, format="decimalyear"),
        "end_time": Time(10, format="decimalyear"),
    }

    rv = {
        "name": "RV",
        "precision": 1.5 * u.m / u.s,
        "rate": 1 / u.d,
        "start_time": Time(0, format="decimalyear"),
        "end_time": Time(5, format="decimalyear"),
    }

    # Save parameters to the builder
    builder.preobs_params = {
        "base_params": base_params,
        "instruments": [eprv, prv, rv],
        "systems_to_observe": [0, 1, 3, 5, 8, 10, 15],
    }
    builder.simulate_rv_observations()

    ######################################################################
    # Orbit fitting
    ######################################################################
    builder.orbit_fitting_params = {"fitting_method": "rvsearch", "max_planets": 3}
    builder.orbit_fitting()
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
