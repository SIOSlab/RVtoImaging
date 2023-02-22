import json
from pathlib import Path

import astropy.units as u

# import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

# import RVtools.plots as plots
from RVtools.builder import BaseBuilder, Director

# from ortools.constraint_solver import pywrapcp, pywraplp, routing_enums_pb2


def create_data_model(coeffs_arr, bounds):
    """Stores the data for the problem."""
    data = {}
    data["constraint_coeffs"] = coeffs_arr
    data["bounds"] = bounds
    data["obj_coeffs"] = np.ones(coeffs_arr.shape[1])
    data["num_constraints"] = coeffs_arr.shape[0]
    data["num_vars"] = coeffs_arr.shape[1]
    return data


class Observation:
    def __init__(self, time, star_ind, star_name, int_time):
        self.time = time
        self.star_ind = star_ind
        self.star_name = star_name
        self.int_time = int_time


if __name__ == "__main__":
    settings_file = Path("/home/corey/Documents/github/AAS_winter_2023/.config.json")
    with open(settings_file, "r") as f:
        settings = json.load(f)
    cache_dir = settings["cache_dir"]
    workers = settings["workers"]

    # Set up director and builder objects
    director = Director()
    builder = BaseBuilder(cache_dir=cache_dir, workers=workers)
    director.builder = builder
    builder.universe_params = {
        "universe_type": "exosims",
        "script": "/home/corey/Documents/github/AAS_winter_2023/caseB.json",
    }
    mission_start = Time(2043, format="decimalyear")
    rv100_25 = {
        "name": "1 m/s",
        "precision": 1 * u.m / u.s,
        "start_time": mission_start - 20 * u.yr,
        "end_time": mission_start - 10 * u.yr,
    }

    rv03_15 = {
        "name": "3 cm/s",
        "precision": 0.03 * u.m / u.s,
        "start_time": mission_start - 10 * u.yr,
    }
    survey = {
        "fit_order": 2,
        "instruments": [rv100_25, rv03_15],
    }
    surveys = [survey]
    base_params = {
        "observation_scheme": "survey",
        "observations_per_night": 10,
        "bad_weather_prob": 0.35,
        "end_time": mission_start,
    }
    nsystems = 100
    builder.preobs_params = {
        "base_params": base_params,
        "surveys": surveys,
        "n_systems_to_observe": nsystems,
        "filters": ["distance"],
    }
    builder.orbitfit_params = {
        "fitting_method": "rvsearch",
        "max_planets": 5,
    }
    construction_method = {"name": "multivariate gaussian", "cov_samples": 1000}
    # construction_method = {"name": "credible interval"}
    builder.pdet_params = {
        "construction_method": construction_method,
        "script": "/home/corey/Documents/github/AAS_winter_2023/caseB.json",
        "number_of_orbits": 500,
        "start_time": mission_start,
        "end_time": mission_start + 5 * u.yr,
    }
    director.build_orbit_fits()
    builder.probability_of_detection()
    builder.schedule_params = {"sim_length": 3 * u.yr, "window_length": 1 * u.d}
    builder.create_schedule()
    # director.run_seeds([0])
    # Making plot
    # results = {}
    # for seed in seeds:
    #     director.run_seeds([seed])
    #     results[seed] = utils.compare_schedule(builder)
