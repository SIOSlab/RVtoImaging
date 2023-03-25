import pickle
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time
from ortools.linear_solver import pywraplp


class ImagingSchedule:
    """
    Base class to do probability of detection calculations
    """

    def __init__(self, params, pdet, universe_dir):
        self.params = params
        self.sim_length = params["sim_length"]
        self.window_length = params["window_length"]
        self.create_schedule(pdet, universe_dir)

    def create_schedule(self, pdet, universe_dir):
        schedule_path = Path(
            universe_dir,
            f"schedule_{self.sim_length}_{self.window_length}.p".replace(" ", ""),
        )
        coeffs_path = Path(
            universe_dir,
            f"coeffs_{self.sim_length}_{self.window_length}.p".replace(" ", ""),
        )
        if schedule_path.exists():
            with open(schedule_path, "rb") as f:
                sorted_observations = pickle.load(f)
            with open(coeffs_path, "rb") as f:
                self.all_coeffs = pickle.load(f)
        else:
            SS = pdet.SS
            start_time = SS.TimeKeeping.missionStart
            end_time = start_time + self.sim_length
            obs_windows = Time(
                np.arange(start_time.jd, end_time.jd, self.window_length.to(u.d).value),
                format="jd",
                scale="tai",
            )
            mode = list(
                filter(
                    lambda mode: mode["detectionMode"],
                    SS.TargetList.OpticalSystem.observingModes,
                )
            )[0]
            self.all_coeffs = {}
            all_orb_sectors = {}
            n_planets = 0
            bounds = []
            koTimes = SS.koTimes
            koMaps = SS.koMaps[mode["syst"]["name"]]
            all_pdets = pdet.pdets
            breakpoint()
            relevant_stars = list(all_pdets.keys())
            for i, star in enumerate(relevant_stars):
                star_name = star.replace("_", " ")
                star_ind = np.where(SS.TargetList.Name == star_name)[0][0]
                star_xr = all_pdets[star].pdet
                # int_times = star_xr.int_time.values
                # obs_times = star_xr.time.values
                star_coeffs = []
                star_orb_sectors = []
                use_star = False
                for planet in star_xr.planet.values:
                    # Get the probability of detection at the desired integration times
                    # and observation times
                    planet_pdet = (
                        all_pdets[star]
                        .pdet.sel(planet=planet)
                        .interp(
                            time=obs_windows.datetime,
                            int_time=self.window_length.to(u.d).value,
                        )
                    )
                    # Splitting up the orbit into sections
                    planet_pop = pdet.pops[star][planet]
                    pop_T = np.median(planet_pop.T)

                    # These orb_sectors separate an orbit into 8 different sectors of
                    # the orbit, which can be used to space out observations for better
                    # orbital coverage
                    orb_sector_times = obs_windows.jd - start_time.jd
                    orb_sectors = (
                        np.arange(
                            0,
                            max(orb_sector_times) + pop_T.to(u.d).value / 8,
                            50,
                        )
                        * u.d
                    )
                    obs_orb_sectors = np.digitize(
                        orb_sector_times, orb_sectors.to(u.d).value
                    )
                    unique_orb_sectors = np.unique(obs_orb_sectors)
                    orb_sector_vals = {}

                    # orb_sectors_to_observe is a orb_sector we can observe,
                    # constraints will be added based on how many there are
                    orb_sectors_to_observe = []
                    orb_sector_maxes = []
                    for orb_sector in unique_orb_sectors:
                        orb_sector_inds = np.where(obs_orb_sectors == orb_sector)
                        # orb_sector_vals[orb_sector] = orb_sector_times[orb_sector_inds
                        orb_sector_vals[orb_sector] = planet_pdet.values[
                            orb_sector_inds
                        ]
                        if np.max(orb_sector_vals[orb_sector]) > 0.5:
                            # orb_sector_maxes.append(np.max(orb_sector_vals[orb_sector]))
                            orb_sector_maxes.append(
                                np.median(orb_sector_vals[orb_sector])
                            )
                        orb_sectors_to_observe.append(orb_sector)
                    # nonzero_pdets = planet_pdet.values[planet_pdet != 0]
                    # median_pdet = np.median(planet_pdet.values)
                    max_pdet = max(
                        sum(orb_sector_maxes[0::2]), sum(orb_sector_maxes[1::2])
                    )
                    bound = np.floor(min(3, max_pdet))
                    if bound > 0.2:
                        print(
                            (
                                f"Adding planet ({pop_T}) around "
                                f"{star_name} with bound {bound}"
                            )
                        )
                        print(f"orb_sector_maxes: {orb_sector_maxes}")
                        # print(f"median_pdet: {median_pdet}")
                        bounds.append(bound)
                        use_star = True
                        n_planets += 1

                        # This interpolates the keepout values to the generated
                        # observation windows
                        obs_window_ko = np.array(
                            np.floor(
                                np.interp(obs_windows.jd, koTimes.jd, koMaps[star_ind])
                            ),
                            dtype=bool,
                        )

                        coeffs = planet_pdet * obs_window_ko
                        star_coeffs.append(coeffs)
                        star_orb_sectors.append(
                            [orb_sectors_to_observe, obs_orb_sectors]
                        )
                    else:
                        print(f"Rejected {star_name} with bound: {bound:.2f}")
                if use_star:
                    self.all_coeffs[star_ind] = np.stack(star_coeffs)
                    all_orb_sectors[star_ind] = star_orb_sectors

            # Create the giant matrix of coefficients
            n_vars = len(self.all_coeffs.keys()) * len(obs_windows)
            coeffs_arr = np.zeros((n_planets, n_vars))
            current_planet = 0
            current_star = 0
            sInd_order = []
            for sInd, coeffs in self.all_coeffs.items():
                next_planet = current_planet + coeffs.shape[0]
                next_star = current_star + 1
                current_col = current_star * coeffs.shape[1]
                next_col = next_star * coeffs.shape[1]
                coeffs_arr[current_planet:next_planet, current_col:next_col] = coeffs
                current_planet = next_planet
                current_star = next_star
                sInd_order.append(sInd)

            solver = pywraplp.Solver.CreateSolver("SCIP")
            data = self.create_data_model(coeffs_arr, bounds)

            # Create the variables, each corresponds to an observation of a star at a
            # specific time
            ind = 0
            x = {}
            all_star_vars = {}
            all_observations = []
            for i, sInd in enumerate(self.all_coeffs.keys()):
                sInd_vars = []
                for j, obs_time in enumerate(obs_windows):
                    all_observations.append(
                        Observation(
                            obs_time,
                            sInd,
                            SS.TargetList.Name[sInd].replace(" ", ""),
                            self.window_length,
                        )
                    )
                    x[ind] = solver.BoolVar(
                        (
                            f"x_[{SS.TargetList.Name[sInd].replace(' ', '')}"
                            f"][{obs_time.datetime}]"
                        )
                    )
                    sInd_vars.append(x[ind])
                    ind += 1
                all_star_vars[sInd] = sInd_vars
            print("Number of variables =", solver.NumVariables())

            # Planet probability of detection constraint
            for i in range(data["num_constraints"]):
                constraint_expr = [
                    data["constraint_coeffs"][i][j] * x[j]
                    for j in range(data["num_vars"])
                ]
                bound = data["bounds"][i]
                solver.Add(sum(constraint_expr) >= bound)

            # Planet sector constraint
            for i, sInd in enumerate(self.all_coeffs.keys()):
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
                            orb_sector_vars = np.array(star_vars)[
                                orb_sector_inds == orb_sector
                            ]
                            next_sector_vars = np.array(star_vars)[
                                orb_sector_inds
                                == (
                                    unique_orb_sectors[
                                        (j + 1) % len(unique_orb_sectors)
                                    ]
                                )
                            ]
                            solver.Add(
                                sum(np.concatenate([orb_sector_vars, next_sector_vars]))
                                <= 1
                            )
                    else:
                        # Otherwise we just want observations in unique orb_sectors
                        for orb_sector in unique_orb_sectors:
                            orb_sector_vars = np.array(star_vars)[
                                orb_sector_inds == orb_sector
                            ]
                            solver.Add(sum(orb_sector_vars) <= 1)

            # Constraint on one star observation per observation time
            for i, obs_time in enumerate(obs_windows):
                vars = [
                    x[j]
                    for j, obs in enumerate(all_observations)
                    if obs.time == obs_time
                ]
                solver.Add(sum(vars) <= 1)

            # Goal of minimizing the number of observations necessary
            obj_expr = [data["obj_coeffs"][j] * x[j] for j in range(data["num_vars"])]
            solver.Minimize(solver.Sum(obj_expr))

            status = solver.Solve()

            observation_list = []
            if status == pywraplp.Solver.OPTIMAL:
                print("Objective value =", solver.Objective().Value())
                for j in range(data["num_vars"]):
                    if x[j].solution_value() > 0:
                        print(x[j].name(), " = ", x[j].solution_value())
                        observation_list.append(all_observations[j])
                print("Problem solved in %f milliseconds" % solver.wall_time())
                print("Problem solved in %d iterations" % solver.iterations())
                print("Problem solved in %d branch-and-bound nodes" % solver.nodes())
            else:
                print("The problem does not have an optimal solution.")

            observation_times = Time(
                [observation.time.jd for observation in observation_list],
                format="jd",
                scale="tai",
            )
            # Sort the observations by time
            sorted_observations = np.array(observation_list)[
                np.argsort(observation_times)
            ]
            with open(schedule_path, "wb") as f:
                pickle.dump(sorted_observations, f)
            with open(coeffs_path, "wb") as f:
                pickle.dump(self.all_coeffs, f)
        self.schedule = sorted_observations

    def create_data_model(self, coeffs_arr, bounds):
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
