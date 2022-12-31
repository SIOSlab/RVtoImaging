import json
from pathlib import Path

import astropy.units as u

# import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from ortools.linear_solver import pywraplp

import RVtools.utils as utils

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


def maxSum(a, n, k):
    if n <= 0:
        return 0
    option = maxSum(a, n - 1, k)
    if k >= a[n - 1]:
        option = max(option, a[n - 1] + maxSum(a, n - 2, k - a[n - 1]))
    return option


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
    def __init__(self, time, star_ind, star_name):
        self.time = time
        self.star_ind = star_ind
        self.star_name = star_name


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
    # n_stars = 1
    # relevant_stars = relevant_stars[:n_stars]
    # relevant_stars = relevant_stars[:7]
    # relevant_stars = [relevant_stars[6]]
    # sim_length = SS.TimeKeeping.missionLife
    sim_length = 365 * u.d
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
    all_orb_sectors = {}
    n_planets = 0
    bounds = []
    for i, star in enumerate(relevant_stars):
        star_name = star.replace("_", " ")
        star_ind = np.where(SS.TargetList.Name == star_name)[0][0]
        star_xr = all_pdets[star].pdet
        int_times = star_xr.int_time.values
        obs_times = star_xr.time.values
        star_coeffs = []
        star_orb_sectors = []
        use_star = False
        for planet in star_xr.planet.values:
            # Get the probability of detection at the desired integration times
            # and observation times
            planet_pdet = (
                all_pdets[star]
                .pdet.sel(planet=planet)
                .interp(time=obs_windows.datetime, int_time=window_length.to(u.d).value)
            )
            # Splitting up the orbit into sections
            planet_pop = builder.rvdata.pdet.pops[star][planet]
            pop_T = np.median(planet_pop.T)

            # These orb_sectors separate an orbit into 8 different sectors of
            # the orbit, which can be used to space out observations for better
            # orbital coverage
            orb_sector_times = obs_windows.jd - start_time.jd
            # orb_sectors = (
            #     np.arange(
            #         0,
            #         max(orb_sector_times) + pop_T.to(u.d).value / 8,
            #         pop_T.to(u.d).value / 8,
            #     )
            #     * u.d
            # )
            orb_sectors = (
                np.arange(
                    0,
                    max(orb_sector_times) + pop_T.to(u.d).value / 8,
                    50,
                )
                * u.d
            )
            obs_orb_sectors = np.digitize(orb_sector_times, orb_sectors.to(u.d).value)
            unique_orb_sectors = np.unique(obs_orb_sectors)
            orb_sector_vals = {}

            # orb_sectors_to_observe is a orb_sector we can observe,
            # constraints will be added based on how many there are
            orb_sectors_to_observe = []
            orb_sector_maxes = []
            for orb_sector in unique_orb_sectors:
                orb_sector_inds = np.where(obs_orb_sectors == orb_sector)
                # orb_sector_vals[orb_sector] = orb_sector_times[orb_sector_inds]
                orb_sector_vals[orb_sector] = planet_pdet.values[orb_sector_inds]
                if np.max(orb_sector_vals[orb_sector]) > 0:
                    # orb_sector_maxes.append(np.max(orb_sector_vals[orb_sector]))
                    orb_sector_maxes.append(np.median(orb_sector_vals[orb_sector]))
                orb_sectors_to_observe.append(orb_sector)
            nonzero_pdets = planet_pdet.values[planet_pdet != 0]
            median_pdet = np.median(planet_pdet.values)
            max_pdet = max(sum(orb_sector_maxes[0::2]), sum(orb_sector_maxes[1::2]))
            bound = np.floor(min(3, max_pdet))
            if bound > 0.5:
                print(
                    f"Adding planet ({pop_T}) around {star_name} with bound of {bound}"
                )
                print(f"orb_sector_maxes: {orb_sector_maxes}")
                # print(f"median_pdet: {median_pdet}")
                bounds.append(bound)
                use_star = True
                n_planets += 1

                # This interpolates the keepout values to the generated
                # observation windows
                obs_window_ko = np.array(
                    np.floor(np.interp(obs_windows.jd, koTimes.jd, koMaps[star_ind])),
                    dtype=bool,
                )

                coeffs = planet_pdet * obs_window_ko
                star_coeffs.append(coeffs)
                star_orb_sectors.append([orb_sectors_to_observe, obs_orb_sectors])
        if use_star:
            all_coeffs[star_ind] = np.stack(star_coeffs)
            all_orb_sectors[star_ind] = star_orb_sectors

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

    solver = pywraplp.Solver.CreateSolver("SCIP")
    data = create_data_model(coeffs_arr, bounds)

    # Create the variables, each corresponds to an observation of a star at a
    # specific time
    ind = 0
    x = {}
    all_star_vars = {}
    observations = []
    for i, sInd in enumerate(all_coeffs.keys()):
        sInd_vars = []
        for j, obs_time in enumerate(obs_windows):
            observations.append(
                Observation(obs_time, sInd, SS.TargetList.Name[sInd].replace(" ", ""))
            )
            x[ind] = solver.BoolVar(
                f"x_[{SS.TargetList.Name[sInd].replace(' ', '')}][{obs_time.datetime}]"
            )
            sInd_vars.append(x[ind])
            ind += 1
        all_star_vars[sInd] = sInd_vars
    print("Number of variables =", solver.NumVariables())

    # Planet probability of detection constraint
    for i in range(data["num_constraints"]):
        constraint_expr = [
            data["constraint_coeffs"][i][j] * x[j] for j in range(data["num_vars"])
        ]
        bound = data["bounds"][i]
        solver.Add(sum(constraint_expr) >= bound)

    # Planet sector constraint
    test_vars = []
    for i, sInd in enumerate(all_coeffs.keys()):
        # Get the constraint variables that correspond to an observation of
        # this planet's star
        star_vars = all_star_vars[sInd]
        for pInd in range(len(all_orb_sectors[sInd])):
            unique_orb_sectors, orb_sector_inds = all_orb_sectors[sInd][pInd]
            if len(unique_orb_sectors) == 1:
                # No need to add this constraint
                continue
            if len(unique_orb_sectors) >= 5:
                # If we have 5 orb_sectors we can set a constraint that we want
                # observations in non-adjacent orb_sectors
                for j, orb_sector in enumerate(unique_orb_sectors):
                    orb_sector_vars = np.array(star_vars)[orb_sector_inds == orb_sector]
                    next_sector_vars = np.array(star_vars)[
                        orb_sector_inds
                        == (unique_orb_sectors[(j + 1) % len(unique_orb_sectors)])
                    ]
                    solver.Add(
                        sum(np.concatenate([orb_sector_vars, next_sector_vars])) <= 1
                    )
            else:
                # Otherwise we just want observations in unique orb_sectors
                for orb_sector in unique_orb_sectors:
                    orb_sector_vars = np.array(star_vars)[orb_sector_inds == orb_sector]
                    solver.Add(sum(orb_sector_vars) <= 1)

    # Constraint on one star observation per observation time
    for i, obs_time in enumerate(obs_windows):
        vars = [x[j] for j, obs in enumerate(observations) if obs.time == obs_time]
        solver.Add(sum(vars) <= 1)

    # Goal of minimizing the number of observations necessary
    obj_expr = [data["obj_coeffs"][j] * x[j] for j in range(data["num_vars"])]
    solver.Minimize(solver.Sum(obj_expr))

    systems = builder.rvdata.universe.systems
    for i, sInd in enumerate(all_coeffs.keys()):
        system_name = SS.TargetList.Name[sInd]
        system = [
            system
            for system in systems
            if system.star.name == system_name.replace(" ", "_")
        ][0]
        SS = utils.replace_EXOSIMS_system(SS, sInd, system)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Objective value =", solver.Objective().Value())
        for j in range(data["num_vars"]):
            if x[j].solution_value() > 0:
                print(x[j].name(), " = ", x[j].solution_value())
        print("Problem solved in %f milliseconds" % solver.wall_time())
        print("Problem solved in %d iterations" % solver.iterations())
        print("Problem solved in %d branch-and-bound nodes" % solver.nodes())
    else:
        print("The problem does not have an optimal solution.")
    # systems = builder.rvdata.universe.systems
    # for i, sInd in enumerate(all_coeffs.keys()):
    #     system_name = SS.TargetList.Name[sInd]
    #     system = [
    #         system
    #         for system in systems
    #         if system.star.name == system_name.replace(" ", "_")
    #     ][0]
    #     if sInd in SS.SimulatedUniverse.plan2star:
    #         SS = utils.replace_EXOSIMS_system(SS, sInd, system)
    # breakpoint()

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
