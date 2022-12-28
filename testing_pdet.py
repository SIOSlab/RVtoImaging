import json
from pathlib import Path

import astropy.units as u

# import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

# import RVtools.plots as plots
from RVtools.builder import BaseBuilder, Director

# from ortools.constraint_solver import pywrapcp, pywraplp, routing_enums_pb2


# def create_sd_mat(sInds, SS):
#     data = {}
#     Obs = SS.Observatory
#     TL = SS.TargetList
#     TK = SS.TimeKeeping
#     mode = list(
#         filter(lambda mode: mode["detectionMode"], TL.OpticalSystem.observingModes)
#     )[0]

#     currentTime = TK.currentTimeAbs.copy() + Obs.settlingTime + mode["syst"]["ohTime"]

#     for old_sInd in sInds:
#         sd = Obs.star_angularSep(TL, old_sInd, sInds, currentTime)
#         obsTimes = Obs.calculate_observableTimes(
#             TL, np.array(sInds), currentTime, SS.koMaps, SS.koTimes, mode
#         )
#         breakpoint()
#         # slewTimes = Obs.calculate_slewTimes(
#         #     TL, old_sInd, sInds, sd, obsTimes, currentTime
#         # )
#     return data


def create_data_model(coeffs_arr, bounds):
    """Stores the data for the problem."""
    data = {}
    data["constraint_coeffs"] = coeffs_arr
    data["bounds"] = bounds
    data["obj_coeffs"] = np.ones(coeffs_arr.shape[1])
    data["num_constraints"] = coeffs_arr.shape[0]
    data["num_vars"] = coeffs_arr.shape[1]
    return data


if __name__ == "__main__":
    # Load settings for this machine
    settings_file = Path(".config.json")
    with open(settings_file, "r") as f:
        settings = json.load(f)
    cache_dir = settings["cache_dir"]
    workers = settings["workers"]
    # first_seed = settings["first_seed"]
    # last_seed = settings["last_seed"]

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
    # base_params = {
    #     "observation_scheme": "survey",
    #     "observations_per_night": 10,
    #     "bad_weather_prob": 0.5,
    #     "end_time": mission_start,
    # }
    # nsystems = 30
    # systems = np.arange(nsystems)
    # builder.preobs_params = {
    #     "base_params": base_params,
    #     "surveys": surveys,
    #     "n_systems_to_observe": nsystems,
    #     "filters": ["distance"],
    # }
    base_params = {
        "observation_scheme": "survey",
        "observations_per_night": 4,
        "bad_weather_prob": 0.5,
        "end_time": mission_start,
    }
    nsystems = 25
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
        "max_planets": 2,
    }

    # RUN THE SEEDS
    seeds = [int(seed) for seed in np.arange(0, 4, 1)]
    ######################################################################
    # Probability of detection
    ######################################################################
    construction_method = {"name": "multivariate gaussian", "cov_samples": 1000}
    # construction_method = {"name": "credible interval"}
    builder.pdet_params = {
        "construction_method": construction_method,
        "script": "scripts/caseA.json",
        "number_of_orbits": 10000,
        "start_time": mission_start,
        "end_time": mission_start + 5 * u.yr,
    }

    director.run_seeds([0])

    # Making plot
    SS = builder.rvdata.pdet.SS
    all_pdets = builder.rvdata.pdet.pdets
    mode = list(
        filter(
            lambda mode: mode["detectionMode"],
            SS.TargetList.OpticalSystem.observingModes,
        )
    )[0]
    koTimes = SS.koTimes
    koMaps = SS.koMaps[mode["syst"]["name"]]
    relevant_stars = list(builder.rvdata.pdet.pdets.keys())
    n_stars = 2
    relevant_stars = relevant_stars[:n_stars]
    # sim_length = SS.TimeKeeping.missionLife
    sim_length = 0.5 * u.yr
    window_length = 0.5 * u.d

    # fig, ax = plt.subplots()
    # ax = plots.init_skyplot(fig, 111)
    sInds = []
    start_time = SS.TimeKeeping.missionStart
    end_time = start_time + sim_length
    obs_windows = Time(
        np.arange(start_time.jd, end_time.jd, window_length.to(u.d).value), format="jd"
    )
    all_coeffs = {}
    n_planets = 0
    bounds = []
    for star in relevant_stars:
        star_name = star.replace("_", " ")
        star_ind = np.where(SS.TargetList.Name == star_name)[0][0]
        # gal_coords = SS.TargetList.coords[star_ind].galactic
        # plots.sky_plot(ax, gal_coords)
        # sInds.append(star_ind)
        star_xr = all_pdets[star].pdet
        int_times = star_xr.int_time.values
        obs_times = star_xr.time.values
        star_coeffs = []
        use_star = False
        for planet in star_xr.planet.values:
            # Get the probability of detection per integration time
            pdet_per_int_time = (
                all_pdets[star].pdet.sel(planet=planet).values.T * 1 / int_times
            ).T
            max_val = np.max(pdet_per_int_time)
            if max_val > 0:
                use_star = True
                n_planets += 1
                max_inds = np.stack(np.where(pdet_per_int_time == max_val)).T
                max_times = Time(obs_times[max_inds[:, 1]])

                # Get the windows with maximum pdet/int_time
                obs_date_diffs = np.diff(Time(obs_times[np.unique(max_inds[:, 1])]).jd)
                breaks = np.argwhere(obs_date_diffs != 1).flatten()

                pdet_window_starts = np.append(
                    max_times[0].jd, max_times[breaks + 1].jd
                )
                pdet_window_ends = np.append(max_times[breaks].jd, max_times[-1].jd)
                pdet_windows = Time(
                    np.stack([pdet_window_starts, pdet_window_ends]).T, format="jd"
                )

                # This is a very tortured line that converts all windows with high
                # probability of detection into a single row of booleans that
                # corresponds to the obs_windows that are used to break up the
                # availble observation times. A True at index 10 means that the
                # planet has high probability of detection at the 10th time in
                # the generated obs_windows.
                obs_window_pdet = np.array(
                    np.sum(
                        [
                            (obs_windows >= pdet_window[0])
                            & (obs_windows <= pdet_window[1])
                            for pdet_window in pdet_windows
                        ],
                        axis=0,
                    ),
                    dtype=bool,
                )

                # This interpolates the keepout values to the generated
                # observation windows
                obs_window_ko = np.array(
                    np.floor(np.interp(obs_windows.jd, koTimes.jd, koMaps[star_ind])),
                    dtype=bool,
                )

                # Combine the two list of bools and multiply by the pdet values to get
                # the coefficients for optimization
                pdet_vals = (
                    all_pdets[star]
                    .pdet.sel(planet=planet)
                    .interp(
                        int_time=window_length.to(u.d).value, time=obs_windows.datetime
                    )
                    .values
                )
                coeffs = pdet_vals * (obs_window_ko & obs_window_pdet)
                star_coeffs.append(coeffs)
                bounds.append(2 * max_val)
        if use_star:
            all_coeffs[star_ind] = np.stack(star_coeffs)

    # Create the giant matrix of coefficients
    n_vars = len(all_coeffs.keys()) * len(obs_windows)
    coeffs_arr = np.zeros((n_planets, n_vars))
    current_planet = 0
    current_star = 0
    sInd_order = []
    for sInd, coeffs in all_coeffs.items():
        next_planet = current_planet + coeffs.shape[0]
        next_star = current_star + 1
        current_col = current_star * coeffs.shape[1]
        next_col = next_star * coeffs.shape[1]
        coeffs_arr[current_planet:next_planet, current_col:next_col] = coeffs
        current_planet = next_planet
        current_star = next_star
        sInd_order.append(sInd)

    # Setting up optimization problem
    # solver = pywraplp.Solver.CreateSolver("SCIP")
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    x = {}
    # ind = 0
    # for i, sInd in enumerate(all_coeffs.keys()):
    #     for j, obs_time in enumerate(obs_windows):
    #         ind += 1
    #         x[ind] = solver.BoolVar(f"x_{sInd},{obs_time.jd}")
    # print("Number of variables =", solver.NumVariables())

    data = create_data_model(coeffs_arr, bounds)
    for j in range(data["num_vars"]):
        x[j] = model.NewBoolVar("x[%i]" % j)
    # print("Number of variables =", model.NumVariables())

    # for i in range(data["num_constraints"]):
    #     constraint = solver.RowConstraint(0, data["bounds"][i], "")
    #     for j in range(data["num_vars"]):
    #         try:
    #             constraint.SetCoefficient(x[j], data["constraint_coeffs"][i][j])
    #         except:
    #             breakpoint()
    for i in range(data["num_constraints"]):
        # constraint_expr = [
        #     data["constraint_coeffs"][i][j] * x[j] for j in range(data["num_vars"])
        # ]
        # model.Add(sum(constraint_expr) >= data["bounds"][i])
        # model.Add(cp_model.LinearExpr.ScalProd(x, data['constraint_coeffs'][i]))
        try:
            model.Add(
                cp_model.LinearExpr.WeightedSum(x, data["constraint_coeffs"][i]) >= 0.01
            )
        except TypeError:
            breakpoint()

    # print("Number of constraints =", model.NumConstraints())

    objective = model.Objective()
    # for j in range(data["num_vars"]):
    #     objective.SetCoefficient(x[j], data["obj_coeffs"][j])
    # objective.SetMinimization()
    obj_expr = [data["obj_coeffs"][j] * x[j] for j in range(data["num_vars"])]
    model.Minimize(model.Sum(obj_expr))

    status = solver.Solve(model)

    if status == pywraplp.Solver.OPTIMAL:
        print("Objective value =", solver.Objective().Value())
        for j in range(data["num_vars"]):
            print(x[j].name(), " = ", x[j].solution_value())
        print()
        print("Problem solved in %f milliseconds" % solver.wall_time())
        print("Problem solved in %d iterations" % solver.iterations())
        print("Problem solved in %d branch-and-bound nodes" % solver.nodes())
    else:
        print("The problem does not have an optimal solution.")

    breakpoint()

    # Need to choose the best pdet/t_int for each observing block
    # data = create_sd_mat(sInds, SS)
    # solver = pywraplp.Solver(
    #     "SolveIntegerProblem", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    # )
    # xs = [solver.IntVar(0.0, 1.0, "x" + str(j)) for j in np.arange(len(compstars))]
    # constraint = solver.Constraint(-solver.infinity(), self.maxTime.to(u.d).value)

    # for j, x in enumerate(xs):
    #     constraint.SetCoefficient(x, tstars[j] + self.ohTimeTot.to(u.day).value)

    # objective = solver.Objective()
    # for j, x in enumerate(xs):
    #     objective.SetCoefficient(x, compstars[j])
    # objective.SetMaximization()
