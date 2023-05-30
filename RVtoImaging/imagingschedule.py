import math
import pickle
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from ortools.sat.python import cp_model
from tqdm import tqdm

from RVtoImaging.logger import logger


class ImagingSchedule:
    """
    Base class to do probability of detection calculations
    """

    def __init__(self, params, pdet, universe_dir, workers):
        self.params = params
        self.sim_length = params["sim_length"]
        self.window_length = params["window_length"]
        self.block_multiples = params["block_multiples"]
        self.max_observations_per_star = params["max_observations_per_star"]
        self.planet_threshold = params["planet_threshold"]
        self.n_observations_above_threshold = params["n_observations_above_threshold"]
        self.time_between_observations = params["time_between_observations"]
        self.log_search_progress = params["log_search_progress"]
        self.max_time_in_seconds = params["max_time_in_seconds"]
        logger.info("Creating Imaging Schedule")
        self.create_schedule(pdet, universe_dir, workers)

    def create_schedule(self, pdet, universe_dir, workers):
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
            start_time_jd = start_time.jd
            end_time = start_time + self.sim_length
            end_time_jd = end_time.jd
            obs_times = Time(
                np.arange(start_time_jd, end_time_jd, self.window_length.to(u.d).value),
                format="jd",
                scale="tai",
            )
            obs_times_datetime = obs_times.datetime
            obs_times_jd = obs_times.jd
            n_obs_times = len(obs_times)
            # int_times = pdet.int_times
            # int_times = pdet.int_times[::3]

            min_int_time = self.window_length.to(u.d).value
            int_times_blocks = self.block_multiples
            int_times = np.array(self.block_multiples) * self.window_length.to(u.d)
            int_times_d = int_times.to(u.d).value
            n_int_times = len(int_times)

            blocks_between_observations = math.ceil(
                self.time_between_observations.to(u.d).value / min_int_time
            )
            time_sector_inds = {}
            inds_per_sector = int(n_obs_times / self.n_time_sectors)
            for i in range(self.n_time_sectors):
                time_sector_inds[i] = np.arange(
                    i * inds_per_sector, (i + 1) * inds_per_sector, 1
                )

            mode = list(
                filter(
                    lambda mode: mode["detectionMode"],
                    SS.TargetList.OpticalSystem.observingModes,
                )
            )[0]
            self.all_coeffs = {}
            koTimes = SS.koTimes
            koTimes_jd = koTimes.jd
            koMaps = SS.koMaps[mode["syst"]["name"]]
            all_pdets = pdet.pdets
            relevant_stars = list(all_pdets.keys())

            # Setting up solver
            star_planets = {}
            model = cp_model.CpModel()
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = workers
            solver.parameters.log_search_progress = self.log_search_progress
            solver.parameters.max_time_in_seconds = self.max_time_in_seconds
            # Coefficients (probability of detections)
            coeffs = {}
            # Dictionary keyed on stars that holds all variables
            self.vars = {}
            all_intervals = []

            # Input sanity checks
            max_schedule_int_time = max(int_times_d)
            max_pdet_int_time = max(all_pdets[relevant_stars[0]].int_time.values)
            assert max_schedule_int_time <= max_pdet_int_time, (
                "Probability of detection cannot be interpolated safely."
                "The highest integration time used to calculate probability of"
                f" detection ({max_pdet_int_time}) must be less than "
                "or equal to and the highest integration time expected to be used for"
                f" scheduling ({max_schedule_int_time})."
            )
            min_schedule_int_time = min(int_times_d)
            min_pdet_int_time = min(all_pdets[relevant_stars[0]].int_time.values)
            assert min_schedule_int_time >= min_pdet_int_time, (
                "Probability of detection cannot be interpolated safely."
                "The lowest integration time used to calculate probability of"
                f" detection ({min_pdet_int_time}) must be greater than "
                "or equal to and the lowest integration time expected to be used for"
                f" scheduling ({min_schedule_int_time})."
            )
            #########################################
            # Stage 1
            #########################################
            logger.info("Creating decision variables and coefficients for optimization")
            for star in tqdm(relevant_stars, desc="stars", position=0):
                # star_name = star.replace("_", " ")
                star_ind = np.where(SS.TargetList.Name == star)[0][0]
                star_xr = all_pdets[star].pdet
                star_planets[star] = star_xr.planet.values
                # Array of booleans for each observation window, for current star
                # True means the star is observable
                obs_window_ko = np.array(
                    np.floor(np.interp(obs_times_jd, koTimes_jd, koMaps[star_ind])),
                    dtype=bool,
                )
                self.vars[star] = {"vars": []}
                for i in range(self.max_observations_per_star):
                    # Creating intervals that can be moved
                    _start = model.NewIntVar(0, len(obs_times) - 1, f"{star} start {i}")
                    _size = model.NewIntVarFromDomain(
                        cp_model.Domain.FromValues(int_times_blocks), f"{star} size {i}"
                    )
                    _end = model.NewIntVar(1, len(obs_times), f"{star} end {i}")
                    _active = model.NewBoolVar(f"{star} active {i}")
                    _interval = model.NewOptionalIntervalVar(
                        _start, _size, _end, _active, f"{star} interval {i}"
                    )
                    all_intervals.append(_interval)
                    self.vars[star]["vars"].append(
                        {
                            "start": _start,
                            "size": _size,
                            "end": _end,
                            "active": _active,
                            "interval": _interval,
                        }
                    )

                # Create the coefficent array, for each observation var we have
                # pdet values for each planet around a star
                star_pdets = np.zeros(
                    (len(star_planets[star]), n_int_times, n_obs_times)
                )
                for planet in star_planets[star]:
                    # shape is (len(int_times), len(obs_times))
                    planet_pdet = np.array(
                        all_pdets[star]
                        .sel(planet=planet)
                        .interp(time=obs_times_datetime, int_time=int_times_d)
                        .pdet
                    )

                    # CPSat solver only works with integers, cast pdet values to
                    # integers by multiplying by 100 and cutting
                    int_planet_pdet = np.array(100 * planet_pdet, dtype=int)
                    # Loop through and set any observations that would go into
                    # keepout to 0
                    for obs_ind, _ in enumerate(obs_times_jd):
                        for int_ind, int_time_d in enumerate(int_times_d):
                            # Number of observing windows this observation spans
                            n_windows = int(int_time_d / min_int_time)

                            # Check that the observation doesn't intersect with keepout
                            ko_vals = obs_window_ko[obs_ind : obs_ind + n_windows]
                            # True if any ko_vals are False
                            in_keepout = ~np.all(ko_vals)
                            if in_keepout:
                                int_planet_pdet[int_ind][obs_ind] = 0
                    star_pdets[planet][:][:] = int_planet_pdet
                self.vars[star]["coeffs"] = star_pdets

                # Set up the planet sectors
                # planet_sectors = np.zeros((len(star_planets[star]), n_obs_times))
                # for planet in star_planets[star]:
                #     planet_pop = pdet.pops[star][planet]
                #     pop_T = np.median(planet_pop.T)

                #     # These orb_sectors separate an orbit into 8 different sectors of
                #     # the orbit, which can be used to space out observations for bette
                #     # orbital coverage
                #     orb_sector_times = obs_times_jd - start_time_jd
                #     obs_orb_sectors = np.digitize(
                #         orb_sector_times % pop_T.to(u.d).value,
                #         np.linspace(0, pop_T.to(u.d).value, 9),
                #     )
                #     unique_orb_sectors = np.unique(obs_orb_sectors)
                #     orb_sector_vals = {}

                # orb_sectors_to_observe is a orb_sector we can observe,
                # constraints will be added based on how many there are
                # orb_sectors_to_observe = []
                # orb_sector_maxes = []
                # for orb_sector in unique_orb_sectors:
                #     orb_sector_inds = np.where(obs_orb_sectors == orb_sector)[0]
                #     orb_sector_vals[orb_sector] = planet_pdet.values[
                #         orb_sector_inds
                #     ]
                #     if np.max(orb_sector_vals[orb_sector]) > self.planet_threshold:
                #         orb_sector_maxes.append(
                #             np.median(orb_sector_vals[orb_sector])
                #         )
                #     orb_sectors_to_observe.append(orb_sector)
                # planet_sectors.append(obs_orb_sectors[obs_ind])

            logger.info("Creating optimization constraints")
            # Constraint on one star observation per observation time
            # for obs_time_jd in tqdm(obs_times_jd, desc="no concurrent observations"):
            #     model.Add(sum(concurrent_obs[obs_time_jd]) <= 1)
            model.AddNoOverlap(all_intervals)

            # Constraint on blocks between observations
            # This is done by looping over all the intervals for each star
            # creating an intermediary boolean that is true when they are both
            # active and adding a constraint that the intervals is above the user
            # defined distance between observations
            for star in relevant_stars:
                star_var_list = self.vars[star]["vars"]
                for i, i_var_set in enumerate(star_var_list[:-1]):
                    # Get model variables
                    i_start = i_var_set["start"]
                    i_end = i_var_set["end"]
                    i_active = i_var_set["active"]
                    for j, j_var_set in enumerate(star_var_list[i + 1 :]):
                        # Get model variables
                        j_start = j_var_set["start"]
                        j_end = j_var_set["end"]
                        j_active = j_var_set["active"]

                        # Create boolean that's active when both intervals are active
                        _bool = model.NewBoolVar(
                            f"{star} blocks between intervals {i} and {j}"
                        )
                        model.Add(i_active + j_active == 2).OnlyEnforceIf(_bool)
                        model.Add(i_active + j_active <= 1).OnlyEnforceIf(_bool.Not())

                        # Bool used to determine which distance we need to be checking
                        i_before_j_bool = model.NewBoolVar(
                            f"{star} interval {i} starts before interval {j}"
                        )
                        model.Add(i_start < j_start).OnlyEnforceIf(i_before_j_bool)
                        model.Add(i_start > j_start).OnlyEnforceIf(
                            i_before_j_bool.Not()
                        )

                        # Add distance constraints
                        model.Add(
                            j_start - i_end >= blocks_between_observations
                        ).OnlyEnforceIf(i_before_j_bool, _bool)
                        model.Add(
                            i_start - j_end >= blocks_between_observations
                        ).OnlyEnforceIf(i_before_j_bool.Not(), _bool)
            # Constraint on total integration time
            # model.Add(sum(int_coeffs) <= len(obs_times))

            # Constraint on observations per star
            # for star in relevant_stars:
            #     model.Add(sum(star_obs[star]) <= 10)

            logger.info("Setting up objective function")
            # Maximize the summed probability of detection
            obj_terms = []
            all_bools = []
            all_active_bools = []
            all_size_vars = []
            planet_terms = {}
            for star in tqdm(relevant_stars, desc="Creating observation booleans"):
                # list of the dictionary of interval variables
                star_var_list = self.vars[star]["vars"]

                # Pdet values for this star
                coeffs = self.vars[star]["coeffs"]

                # Dictionary keyed on [star][planet][sector] that gets used to
                # create the constraint for the planets
                planet_terms[star] = {}
                for sector in range(self.n_time_sectors):
                    planet_terms[star][sector] = {}
                    for planet in star_planets[star]:
                        planet_terms[star][sector][planet] = []

                # Loop over all the intervals, defined by start, size, end,
                # active, for this star
                for i, var_set in enumerate(star_var_list):
                    start_var = var_set["start"]
                    size_var = var_set["size"]
                    active_var = var_set["active"]
                    all_active_bools.append(active_var)
                    all_size_vars.append(size_var)
                    set_bools = []
                    for n_int, n_times_blocks in enumerate(int_times_blocks):
                        for n_obs in range(n_obs_times):
                            if n_obs in time_sector_inds[i]:
                                _bool = model.NewBoolVar(f"{star} {n_int} {n_obs}")
                                set_bools.append(_bool)
                                model.Add(start_var == n_obs).OnlyEnforceIf(_bool)
                                model.Add(size_var == n_times_blocks).OnlyEnforceIf(
                                    _bool
                                )
                                model.Add(active_var == 1).OnlyEnforceIf(_bool)
                                obj_terms.append(_bool * sum(coeffs[:, n_int, n_obs]))
                                for planet in star_planets[star]:
                                    planet_terms[star][i][planet].append(
                                        (_bool, coeffs[planet, n_int, n_obs])
                                    )
                    var_set["bools"] = set_bools
                    all_bools.extend(set_bools)

            # Set up planet constraint, number of observations above 0.5 pdet
            # are less than 4
            # for star in relevant_stars:
            #     for planet in star_planets[star]:
            #         # planets_above_threshold.append(
            #         #     model.NewBoolVar(f"{star}{planet} meets threshold")
            #         # )
            #         above_threshold_val = []
            #         for sector in range(self.n_time_sectors):
            #             for _bool, coeff in planet_terms[star][sector][planet]:
            #                 if coeff > self.planet_threshold:
            #                     above_threshold_val.append(_bool)
            #         model.Add(sum(above_threshold_val) <= self.max_planet_observations
            # model.Maximize(sum(obj_terms))
            planets_above_threshold = []
            for star in relevant_stars:
                for planet in star_planets[star]:
                    this_planet_bool = model.NewBoolVar(
                        f"{star}{planet} meets threshold"
                    )
                    planets_above_threshold.append(this_planet_bool)
                    above_threshold_val = []
                    for sector in range(self.n_time_sectors):
                        for _bool, coeff in planet_terms[star][sector][planet]:
                            if coeff > 100 * self.planet_threshold:
                                above_threshold_val.append(_bool)
                    model.Add(
                        sum(above_threshold_val) >= self.n_observations_above_threshold
                    ).OnlyEnforceIf(this_planet_bool)
            model.Maximize(
                sum(100 * max(self.block_multiples) * planets_above_threshold)
                - sum(all_size_vars)
                - sum(all_active_bools)
            )

            logger.info("Running optimization solver")
            solver.Solve(model)

            planet_ub = max([len(item) for _, item in star_planets.items()])
            observation_list = []
            final_stars = []
            final_times = []
            final_int_times = []
            total_coeffs = []
            planet_thresholds = {}
            planet_coeffs = {}
            for pnum in range(planet_ub):
                planet_coeffs[pnum] = []
                planet_thresholds[pnum] = []
            # Create dataframe of the observations
            for star in relevant_stars:
                star_var_list = self.vars[star]["vars"]
                coeffs = self.vars[star]["coeffs"]
                for i, var_set in enumerate(star_var_list):
                    start_var = var_set["start"]
                    size_var = var_set["size"]
                    active_var = var_set["active"]
                    if solver.Value(active_var):
                        final_stars.append(star)
                        n_obs = solver.Value(start_var)
                        n_int = solver.Value(size_var)
                        n_int_block = np.where(n_int == np.array(int_times_blocks))[0][
                            0
                        ]
                        final_times.append(obs_times[n_obs])
                        final_int_times.append(int_times[n_int_block])
                        above_threshold = False
                        summed_coeff = 0
                        for pnum in range(planet_ub):
                            if pnum in star_planets[star]:
                                coeff = coeffs[pnum, n_int_block, n_obs]
                                summed_coeff += coeff
                                above_threshold = coeff > 100 * self.planet_threshold
                            else:
                                coeff = np.nan
                                above_threshold = np.nan
                            planet_coeffs[pnum].append(coeff)
                            planet_thresholds[pnum].append(above_threshold)

                        total_coeffs.append(summed_coeff)

            df = pd.DataFrame()
            df["star"] = final_stars
            df["time"] = final_times
            df["int_time"] = final_int_times
            df["total_coeffs"] = total_coeffs
            for pnum in range(planet_ub):
                df[f"coeff{pnum}"] = planet_coeffs[pnum]
            for pnum in range(planet_ub):
                df[f"threshold{pnum}"] = planet_thresholds[pnum]

            # Calculations for the number of detections for each
            unique_planets_above_threshold = 0
            planets_fitted = 0
            planet_data = {}
            for star in relevant_stars:
                for planet in star_planets[star]:
                    planet_observations = df.loc[df.star == star][
                        f"threshold{planet}"
                    ].values
                    if len(planet_observations):
                        n_above_threshold = sum(planet_observations)
                        planet_data[planets_fitted] = {
                            "star": star,
                            "planet": planet,
                            "n_above_threshold": n_above_threshold,
                        }
                        if n_above_threshold:
                            unique_planets_above_threshold += 1
                    planets_fitted += 1
            pd.DataFrame(planet_data).T
            breakpoint()

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

    # def create_schedule_test1(self, pdet, universe_dir):
    #     schedule_path = Path(
    #         universe_dir,
    #         f"schedule_{self.sim_length}_{self.window_length}.p".replace(" ", ""),
    #     )
    #     coeffs_path = Path(
    #         universe_dir,
    #         f"coeffs_{self.sim_length}_{self.window_length}.p".replace(" ", ""),
    #     )
    #     if schedule_path.exists():
    #         with open(schedule_path, "rb") as f:
    #             sorted_observations = pickle.load(f)
    #         with open(coeffs_path, "rb") as f:
    #             self.all_coeffs = pickle.load(f)
    #     else:
    #         SS = pdet.SS
    #         start_time = SS.TimeKeeping.missionStart
    #         start_time_jd = start_time.jd
    #         end_time = start_time + self.sim_length
    #         end_time_jd = end_time.jd
    #         obs_times = Time(
    #             np.arange(start_time.jd, end_time_jd, self.window_length.to(u.d).value
    #             format="jd",
    #             scale="tai",
    #         )
    #         obs_times_datetime = obs_times.datetime
    #         obs_times_jd = obs_times.jd
    #         # int_times = pdet.int_times
    #         int_times = pdet.int_times[::3]
    #         int_times_d = int_times.to(u.d).value

    #         min_int_time = int_times[0].to(u.d).value
    #         mode = list(
    #             filter(
    #                 lambda mode: mode["detectionMode"],
    #                 SS.TargetList.OpticalSystem.observingModes,
    #             )
    #         )[0]
    #         self.all_coeffs = {}
    #         all_orb_sectors = {}
    #         n_planets = 0
    #         bounds = []
    #         koTimes = SS.koTimes
    #         koTimes_jd = koTimes.jd
    #         koMaps = SS.koMaps[mode["syst"]["name"]]
    #         all_pdets = pdet.pdets
    #         relevant_stars = list(all_pdets.keys())

    #         # Setting up solver
    #         star_planets = {}
    #         # solver = pywraplp.Solver.CreateSolver("SCIP")
    #         model = cp_model.CpModel()
    #         solver = cp_model.CpSolver()
    #         solver.parameters.num_search_workers = 10
    #         solver.parameters.log_search_progress = True
    #         # solver.parameters.max_time_in_seconds = 360
    #         # Decision variables
    #         # observation_objs = []
    #         too_long = {}
    #         in_keepout = {}
    #         observation_vars = {}
    #         # Coefficients (probability of detections)
    #         coeffs = {}
    #         # Integration time coefficients
    #         int_coeffs = []
    #         # Dictionary keyed on observation times that holds all observation
    #         # variables that overlap with that observation time
    #         # concurrent_obs = {}
    #         # for obs_time in obs_times:
    #         #     concurrent_obs[obs_time.jd] = []
    #         interval_vars = []
    #         # Dict keyed on stars for max observation constraint
    #         star_obs = {}
    #         for star in relevant_stars:
    #             star_obs[star] = []

    #         logger.info("Creating decision variables and coefficients for optimization
    #         for star in tqdm(relevant_stars, desc="stars", position=0):
    #             # star_name = star.replace("_", " ")
    #             star_ind = np.where(SS.TargetList.Name == star)[0][0]
    #             star_xr = all_pdets[star].pdet
    #             star_orb_sectors = []
    #             star_planets[star] = star_xr.planet.values
    #             # Array of booleans for each observation window, for current star
    #             # True means the star is observable
    #             obs_window_ko = np.array(
    #                 np.floor(np.interp(obs_times_jd, koTimes_jd, koMaps[star_ind])),
    #                 dtype=bool,
    #             )
    #             for obs_ind, obs_time_jd in enumerate(obs_times_jd):
    #                 for int_time_d in int_times_d:
    #                     # Time this observation would end
    #                     obs_end = obs_time_jd + int_time_d

    #                     # Number of observing windows this observation spans
    #                     n_windows = int(int_time_d / min_int_time)

    #                     # Check that the observation ends within the alloted time
    #                     # e.g. if we have 365 days we can't start a 40 day observation
    #                     # on day 364
    #                     if obs_end > end_time_jd:
    #                         too_long[(star, obs_time_jd, int_time_d)] = Observation(
    #                             star, obs_time_jd, int_time_d
    #                         )
    #                         continue

    #                     # Check that the observation doesn't intersect with keepout
    #                     ko_vals = obs_window_ko[obs_ind : obs_ind + n_windows]
    #                     # False if any ko_vals are False
    #                     keepout_good = np.all(ko_vals)
    #                     if not keepout_good:
    #                         in_keepout[(star, obs_time_jd, int_time_d)] = Observation(
    #                             star, obs_time_jd, int_time_d
    #                         )
    #                         continue

    #                     _var = model.NewBoolVar(
    #                         f"{star}, {obs_times_datetime[obs_ind]}, {int_time_d:.2f}"
    #                     )
    #                     observation_vars[(star, obs_time_jd, int_time_d)] = _var

    #                     # Add var to concurrent_obs dict in all intersecting time bloc
    #                     interval_vars.append(
    #                         model.NewOptionalIntervalVar(
    #                             obs_ind,
    #                             n_windows,
    #                             obs_ind + n_windows,
    #                             _var,
    #                             f"{star}, {obs_times_datetime[obs_ind]}, {int_time_d:.
    #                         )
    #                     )
    #                     # cond1 = obs_time_jd <= obs_times_jd
    #                     # cond2 = (obs_time_jd + int_time_d) > obs_times_jd
    #                     # time_blocks_this_var = obs_times_jd[cond1 & cond2]
    #                     # for time_block_this_var in time_blocks_this_var:
    #                     #     concurrent_obs[time_block_this_var].append(_var)

    #                     # Integration time coefficients
    #                     # int_coeffs.append(int_time_d * _var)
    #                     int_coeffs.append(n_windows * _var)

    #                     star_obs[star].append(_var)

    #             # Create the coefficent array, for each observation var we have
    #             # pdet values for each planet around a star
    #             for planet in star_planets[star]:
    #                 # shape is (len(int_times), len(obs_times))
    #                 planet_pdet = np.array(
    #                     all_pdets[star]
    #                     .sel(planet=planet)
    #                     .interp(time=obs_times_datetime, int_time=int_times)
    #                     .pdet
    #                 )
    #                 # .sel(planet=planet, time=obs_times_datetime, int_time=int_times)

    #                 # # Splitting up the orbit into sections
    #                 # planet_pop = pdet.pops[star][planet]
    #                 # pop_T = np.median(planet_pop.T.to(u.d).value)

    #                 # # These orb_sectors separate an orbit into 8 different sectors o
    #                 # # the orbit, which can be used to space out observations for bet
    #                 # # orbital coverage
    #                 # orb_sector_times = obs_times_jd - start_time_jd
    #                 # orb_sectors = np.arange(
    #                 #     0,
    #                 #     max(orb_sector_times) + pop_T / 8,
    #                 #     50,
    #                 # )
    #                 # obs_orb_sectors = np.digitize(orb_sector_times, orb_sectors)
    #                 # unique_orb_sectors = np.unique(obs_orb_sectors)
    #                 # orb_sector_vals = {}

    #                 # # orb_sectors_to_observe is a orb_sector we can observe,
    #                 # # constraints will be added based on how many there are
    #                 # orb_sectors_to_observe = []
    #                 # orb_sector_maxes = []
    #                 # min_int_pdet = planet_pdet[0]
    #                 # for orb_sector in unique_orb_sectors:
    #                 #     orb_sector_inds = np.where(obs_orb_sectors == orb_sector)
    #                 #     # orb_sector_vals[orb_sector]= orb_sector_times[orb_sector_i
    #                 #     orb_sector_vals[orb_sector] = min_int_pdet[orb_sector_inds]
    #                 #     if np.max(orb_sector_vals[orb_sector]) > 0.5:
    #                 #         # orb_sector_maxes.append(np.max(orb_sector_vals[orb_sec
    #                 #         orb_sector_maxes.append(
    #                 #             np.median(orb_sector_vals[orb_sector])
    #                 #         )
    #                 #     orb_sectors_to_observe.append(orb_sector)
    #                 # breakpoint()

    #                 # CPSat solver only works with integers, cast pdet values to
    #                 # integers by multiplying by 10000 and cutting
    #                 int_planet_pdet = np.array(10000 * planet_pdet, dtype=int)
    #                 obs_vars_keys = observation_vars.keys()
    #                 for obs_ind, obs_time_jd in enumerate(obs_times_jd):
    #                     for int_ind, int_time_d in enumerate(int_times_d):
    #                         # This if statement makes sure that the observation is val
    #                         if (star, obs_time_jd, int_time_d) in obs_vars_keys:
    #                             # obs_pdet = planet_pdet[int_ind, obs_ind]
    #                             obs_pdet = int_planet_pdet[int_ind, obs_ind]
    #                             coeffs[
    #                                 (
    #                                     star,
    #                                     planet,
    #                                     obs_time_jd,
    #                                     int_time_d,
    #                                 )
    #                             ] = obs_pdet

    #         logger.info("Creating optimization constraints")
    #         # Constraint on one star observation per observation time
    #         # for obs_time_jd in tqdm(obs_times_jd, desc="no concurrent observations")
    #         #     model.Add(sum(concurrent_obs[obs_time_jd]) <= 1)
    #         model.AddNoOverlap(interval_vars)

    #         # Constraint on total integration time
    #         # model.Add(sum(int_coeffs) <= len(obs_times))

    #         # Constraint on observations per star
    #         # for star in relevant_stars:
    #         #     model.Add(sum(star_obs[star]) <= 10)

    #         logger.info("Setting up objective function")
    #         # Maximize the summed probability of detection
    #         obj_terms = []
    #         for obs_key, obs_var in observation_vars.items():
    #             star, obs_time_jd, int_time_d = obs_key
    #             for planet in star_planets[star]:
    #                 obj_terms.append(
    #                     obs_var * coeffs[(star, planet, obs_time_jd, int_time_d)]
    #                 )

    #         # obj_expr = [data["obj_coeffs"][j] * x[j] for j in range(data["num_vars"]
    #         model.Maximize(sum(obj_terms))

    #         logger.info("Running optimization solver")
    #         status = solver.Solve(model)

    #         observation_list = []
    #         final_stars = []
    #         final_times = []
    #         final_int_times = []
    #         final_coeffs = []
    #         for obs_key, obs_var in observation_vars.items():
    #             if solver.Value(obs_var):
    #                 observation_list.append(obs_var)
    #                 star, obs_time_jd, int_time_d = obs_key
    #                 final_stars.append(star)
    #                 final_times.append(Time(obs_time_jd, format="jd"))
    #                 final_int_times.append(int_time_d)
    #                 obs_coeff = 0
    #                 for planet in star_planets[star]:
    #                     obs_coeff += coeffs[(star, planet, obs_time_jd, int_time_d)]
    #                 final_coeffs.append(obs_coeff)
    #         df = pd.DataFrame()
    #         df["star"] = final_stars
    #         df["time"] = final_times
    #         df["int_time"] = final_int_times
    #         df["coeff"] = final_coeffs
    #         breakpoint()
    #         # if status == pywraplp.Solver.OPTIMAL:
    #         #     print("Objective value =", solver.Objective().Value())
    #         #     for j in range(data["num_vars"]):
    #         #         if x[j].solution_value() > 0:
    #         #             print(x[j].name(), " = ", x[j].solution_value())
    #         #             observation_list.append(all_observations[j])
    #         #     print("Problem solved in %f milliseconds" % solver.wall_time())
    #         #     print("Problem solved in %d iterations" % solver.iterations())
    #         #     print("Problem solved in %d branch-and-bound nodes" % solver.nodes()
    #         # else:
    #         #     print("The problem does not have an optimal solution.")

    #         observation_times = Time(
    #             [observation.time.jd for observation in observation_list],
    #             format="jd",
    #             scale="tai",
    #         )
    #         # Sort the observations by time
    #         sorted_observations = np.array(observation_list)[
    #             np.argsort(observation_times)
    #         ]
    #         with open(schedule_path, "wb") as f:
    #             pickle.dump(sorted_observations, f)
    #         with open(coeffs_path, "wb") as f:
    #             pickle.dump(self.all_coeffs, f)
    #     self.schedule = sorted_observations

    # def create_schedule_original(self, pdet, universe_dir):
    #     schedule_path = Path(
    #         universe_dir,
    #         f"schedule_{self.sim_length}_{self.window_length}.p".replace(" ", ""),
    #     )
    #     coeffs_path = Path(
    #         universe_dir,
    #         f"coeffs_{self.sim_length}_{self.window_length}.p".replace(" ", ""),
    #     )
    #     if schedule_path.exists():
    #         with open(schedule_path, "rb") as f:
    #             sorted_observations = pickle.load(f)
    #         with open(coeffs_path, "rb") as f:
    #             self.all_coeffs = pickle.load(f)
    #     else:
    #         SS = pdet.SS
    #         start_time = SS.TimeKeeping.missionStart
    #         end_time = start_time + self.sim_length
    #         obs_times = Time(
    #             np.arange(start_time.jd, end_time.jd,
    #                        self.window_length.to(u.d).value),
    #             format="jd",
    #             scale="tai",
    #         )
    #         mode = list(
    #             filter(
    #                 lambda mode: mode["detectionMode"],
    #                 SS.TargetList.OpticalSystem.observingModes,
    #             )
    #         )[0]
    #         self.all_coeffs = {}
    #         all_orb_sectors = {}
    #         n_planets = 0
    #         bounds = []
    #         koTimes = SS.koTimes
    #         koMaps = SS.koMaps[mode["syst"]["name"]]
    #         all_pdets = pdet.pdets
    #         relevant_stars = list(all_pdets.keys())
    #         for i, star in enumerate(relevant_stars):
    #             star_name = star.replace("_", " ")
    #             star_ind = np.where(SS.TargetList.Name == star_name)[0][0]
    #             star_xr = all_pdets[star].pdet
    #             # int_times = star_xr.int_time.values
    #             # obs_times = star_xr.time.values
    #             star_coeffs = []
    #             star_orb_sectors = []
    #             use_star = False
    #             for planet in star_xr.planet.values:
    #                 # Get the probability of detection at the desired integration time
    #                 # and observation times
    #                 breakpoint()
    #                 planet_pdet = (
    #                     all_pdets[star]
    #                     .pdet.sel(planet=planet)
    #                     .interp(
    #                         time=obs_times.datetime,
    #                         int_time=self.window_length.to(u.d).value,
    #                     )
    #                 )
    #                 # Splitting up the orbit into sections
    #                 planet_pop = pdet.pops[star][planet]
    #                 pop_T = np.median(planet_pop.T)

    #                 # These orb_sectors separate an orbit into 8 different sectors of
    #                 # the orbit, which can be used to space out observations for bette
    #                 # orbital coverage
    #                 orb_sector_times = obs_times.jd - start_time.jd
    #                 orb_sectors = (
    #                     np.arange(
    #                         0,
    #                         max(orb_sector_times) + pop_T.to(u.d).value / 8,
    #                         50,
    #                     )
    #                     * u.d
    #                 )
    #                 obs_orb_sectors = np.digitize(
    #                     orb_sector_times, orb_sectors.to(u.d).value
    #                 )
    #                 unique_orb_sectors = np.unique(obs_orb_sectors)
    #                 orb_sector_vals = {}

    #                 # orb_sectors_to_observe is a orb_sector we can observe,
    #                 # constraints will be added based on how many there are
    #                 orb_sectors_to_observe = []
    #                 orb_sector_maxes = []
    #                 for orb_sector in unique_orb_sectors:
    #                     orb_sector_inds = np.where(obs_orb_sectors == orb_sector)
    #                     # orb_sector_vals[orb_sector] = orb_sector_times[
    #        orb_sector_inds
    #                     orb_sector_vals[orb_sector] = planet_pdet.values[
    #                         orb_sector_inds
    #                     ]
    #                     if np.max(orb_sector_vals[orb_sector]) > 0.5:
    #                         # orb_sector_maxes.append(np.max(orb_sector_vals[
    #    orb_sector]))
    #                         orb_sector_maxes.append(
    #                             np.median(orb_sector_vals[orb_sector])
    #                         )
    #                     orb_sectors_to_observe.append(orb_sector)
    #                 # nonzero_pdets = planet_pdet.values[planet_pdet != 0]
    #                 # median_pdet = np.median(planet_pdet.values)
    #                 max_pdet = max(
    #                     sum(orb_sector_maxes[0::2]), sum(orb_sector_maxes[1::2])
    #                 )
    #                 bound = np.floor(min(3, max_pdet))
    #                 if bound > 0.2:
    #                     print(
    #                         (
    #                             f"Adding planet ({pop_T}) around "
    #                             f"{star_name} with bound {bound}"
    #                         )
    #                     )
    #                     print(f"orb_sector_maxes: {orb_sector_maxes}")
    #                     # print(f"median_pdet: {median_pdet}")
    #                     bounds.append(bound)
    #                     use_star = True
    #                     n_planets += 1

    #                     # This interpolates the keepout values to the generated
    #                     # observation windows
    #                     obs_window_ko = np.array(
    #                         np.floor(
    #                             np.interp(obs_times.jd, koTimes.jd, koMaps[star_ind])
    #                         ),
    #                         dtype=bool,
    #                     )

    #                     coeffs = planet_pdet * obs_window_ko
    #                     star_coeffs.append(coeffs)
    #                     star_orb_sectors.append(
    #                         [orb_sectors_to_observe, obs_orb_sectors]
    #                     )
    #                 else:
    #                     print(f"Rejected {star_name} with bound: {bound:.2f}")
    #             if use_star:
    #                 self.all_coeffs[star_ind] = np.stack(star_coeffs)
    #                 all_orb_sectors[star_ind] = star_orb_sectors

    #         # Create the giant matrix of coefficients
    #         n_vars = len(self.all_coeffs.keys()) * len(obs_times)
    #         coeffs_arr = np.zeros((n_planets, n_vars))
    #         current_planet = 0
    #         current_star = 0
    #         sInd_order = []
    #         for sInd, coeffs in self.all_coeffs.items():
    #             next_planet = current_planet + coeffs.shape[0]
    #             next_star = current_star + 1
    #             current_col = current_star * coeffs.shape[1]
    #             next_col = next_star * coeffs.shape[1]
    #             coeffs_arr[current_planet:next_planet, current_col:next_col] = coeffs
    #             current_planet = next_planet
    #             current_star = next_star
    #             sInd_order.append(sInd)

    #         solver = pywraplp.Solver.CreateSolver("SCIP")
    #         data = self.create_data_model(coeffs_arr, bounds)

    #         # Create the variables, each corresponds to an observation of a star at a
    #         # specific time
    #         ind = 0
    #         x = {}
    #         all_star_vars = {}
    #         all_observations = []
    #         for i, sInd in enumerate(self.all_coeffs.keys()):
    #             sInd_vars = []
    #             for j, obs_time in enumerate(obs_times):
    #                 all_observations.append(
    #                     Observation(
    #                         obs_time,
    #                         sInd,
    #                         SS.TargetList.Name[sInd].replace(" ", ""),
    #                         self.window_length,
    #                     )
    #                 )
    #                 x[ind] = solver.BoolVar(
    #                     (
    #                         f"x_[{SS.TargetList.Name[sInd].replace(' ', '')}"
    #                         f"][{obs_time.datetime}]"
    #                     )
    #                 )
    #                 sInd_vars.append(x[ind])
    #                 ind += 1
    #             all_star_vars[sInd] = sInd_vars
    #         print("Number of variables =", solver.NumVariables())

    #         # Planet probability of detection constraint
    #         for i in range(data["num_constraints"]):
    #             constraint_expr = [
    #                 data["constraint_coeffs"][i][j] * x[j]
    #                 for j in range(data["num_vars"])
    #             ]
    #             bound = data["bounds"][i]
    #             solver.Add(sum(constraint_expr) >= bound)

    #         # Planet sector constraint
    #         for i, sInd in enumerate(self.all_coeffs.keys()):
    #             # Get the constraint variables that correspond to an observation of
    #             # this planet's star
    #             star_vars = all_star_vars[sInd]
    #             for pInd in range(len(all_orb_sectors[sInd])):
    #                 unique_orb_sectors, orb_sector_inds = all_orb_sectors[sInd][pInd]
    #                 if len(unique_orb_sectors) == 1:
    #                     # No need to add this constraint
    #                     continue
    #                 if len(unique_orb_sectors) >= 5:
    #                     # If we have 5 orb_sectors we can set a constraint that we wan
    #                     # observations in non-adjacent orb_sectors
    #                     for j, orb_sector in enumerate(unique_orb_sectors):
    #                         orb_sector_vars = np.array(star_vars)[
    #                             orb_sector_inds == orb_sector
    #                         ]
    #                         next_sector_vars = np.array(star_vars)[
    #                             orb_sector_inds
    #                             == (
    #                                 unique_orb_sectors[
    #                                     (j + 1) % len(unique_orb_sectors)
    #                                 ]
    #                             )
    #                         ]
    #                         solver.Add(
    #                             sum(np.concatenate([orb_sector_vars, next_sector_vars]
    #                             <= 1
    #                         )
    #                 else:
    #                     # Otherwise we just want observations in unique orb_sectors
    #                     for orb_sector in unique_orb_sectors:
    #                         orb_sector_vars = np.array(star_vars)[
    #                             orb_sector_inds == orb_sector
    #                         ]
    #                         solver.Add(sum(orb_sector_vars) <= 1)

    #         # Constraint on one star observation per observation time
    #         for i, obs_time in enumerate(obs_times):
    #             vars = [
    #                 x[j]
    #                 for j, obs in enumerate(all_observations)
    #                 if obs.time == obs_time
    #             ]
    #             solver.Add(sum(vars) <= 1)

    #         # Goal of minimizing the number of observations necessary
    #         obj_expr = [data["obj_coeffs"][j] * x[j] for j in range(data["num_vars"])]
    #         solver.Minimize(solver.Sum(obj_expr))

    #         status = solver.Solve()

    #         observation_list = []
    #         if status == pywraplp.Solver.OPTIMAL:
    #             print("Objective value =", solver.Objective().Value())
    #             for j in range(data["num_vars"]):
    #                 if x[j].solution_value() > 0:
    #                     print(x[j].name(), " = ", x[j].solution_value())
    #                     observation_list.append(all_observations[j])
    #             print("Problem solved in %f milliseconds" % solver.wall_time())
    #             print("Problem solved in %d iterations" % solver.iterations())
    #             print("Problem solved in %d branch-and-bound nodes" % solver.nodes())
    #         else:
    #             print("The problem does not have an optimal solution.")

    #         observation_times = Time(
    #             [observation.time.jd for observation in observation_list],
    #             format="jd",
    #             scale="tai",
    #         )
    #         # Sort the observations by time
    #         sorted_observations = np.array(observation_list)[
    #             np.argsort(observation_times)
    #         ]
    #         with open(schedule_path, "wb") as f:
    #             pickle.dump(sorted_observations, f)
    #         with open(coeffs_path, "wb") as f:
    #             pickle.dump(self.all_coeffs, f)
    #     self.schedule = sorted_observations

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
    def __init__(self, star, obs_time, int_time):
        self.star = star
        self.obs_time = obs_time
        self.int_time = int_time
