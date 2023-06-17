import copy
import itertools
import math
import pickle
from collections import Counter
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from ortools.sat.python import cp_model
from tqdm import tqdm

from EXOSIMS.util.utils import dictToSortedStr, genHexStr
from RVtoImaging.logger import logger


class ImagingSchedule:
    """
    Base class to do probability of detection calculations
    """

    def __init__(self, params, rv_dataset_params, pdet, universe_dir, workers):
        params["workers"] = workers
        params["pdet_hash"] = pdet.settings_hash
        params["rv_dataset"] = {}
        self.best_precision = np.inf
        for obs_run in rv_dataset_params["rv_observing_runs"]:
            # This is used to guarantee we're using a hash that a
            simple_run_spec = {}
            simple_run_spec["location"] = obs_run["location"]
            simple_run_spec["start_time"] = obs_run["start_time"].jd
            simple_run_spec["end_time"] = obs_run["end_time"].jd
            simple_run_spec["sigma_terms"] = obs_run["sigma_terms"]

            sigmas = np.array(
                [val.to(u.m / u.s).value for val in obs_run["sigma_terms"].values()]
            )
            run_rms = np.sqrt(np.mean(np.square(sigmas)))
            if run_rms < self.best_precision:
                self.best_precision = run_rms
            scheme_spec = {}
            for scheme_term in obs_run["observation_scheme"].keys():
                if scheme_term != "astroplan_constraints":
                    scheme_spec[scheme_term] = obs_run["observation_scheme"][
                        scheme_term
                    ]
            simple_run_spec["observation_scheme"] = scheme_spec

            name = obs_run["name"]
            params["rv_dataset"][name] = simple_run_spec

        params["rv_dataset_name"] = rv_dataset_params["dataset_name"]
        self.params = params
        self.coeff_multiple = params["coeff_multiple"]
        self.sim_length = params["sim_length"]
        self.block_length = params["block_length"]
        self.block_multiples = params["block_multiples"]
        self.max_observations_per_star = params["max_observations_per_star"]
        self.planet_threshold = params["planet_threshold"]
        self.requested_planet_observations = params["requested_planet_observations"]
        self.min_required_wait_time = params["min_required_wait_time"]
        self.max_required_wait_time = params["max_required_wait_time"]
        self.log_search_progress = params["log_search_progress"]
        self.max_time_in_seconds = params["max_time_in_seconds"]
        self.hash = genHexStr(dictToSortedStr(params))

        self.result_path = Path(universe_dir, "results", f"schedule_{self.hash}")
        self.result_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Creating Imaging Schedule {self.hash}"
            f" for {universe_dir.parts[-1].replace('_', ' ')} "
        )
        self.create_schedule(pdet, universe_dir, workers)
        # Add schedule to the SS module
        if self.schedule.empty:
            logger.warn(f"No observations scheduled for {self.result_path}")
        else:
            with open(Path(self.result_path, "spec.p"), "wb") as f:
                pickle.dump(self.params, f)

            target_info_path = Path(self.result_path, "per_target.p")
            flat_info_path = Path(self.result_path, "flat_info.p")
            summary_info_path = Path(self.result_path, "summary.p")
            if not target_info_path.exists():
                (
                    self.targetdf,
                    self.flatdf,
                    self.summary_stats,
                ) = pdet.SS.sim_fixed_schedule(self.schedule)
                self.summary_stats["fitted_planets"] = sum(
                    [len(pdet.pops[key]) for key in pdet.pops.keys()]
                )
                with open(target_info_path, "wb") as f:
                    pickle.dump(self.targetdf, f)
                with open(flat_info_path, "wb") as f:
                    pickle.dump(self.flatdf, f)
                with open(summary_info_path, "wb") as f:
                    pickle.dump(self.summary_stats, f)
            else:
                with open(target_info_path, "rb") as f:
                    self.targetdf = pickle.load(f)
                with open(flat_info_path, "rb") as f:
                    self.flatdf = pickle.load(f)
                with open(summary_info_path, "rb") as f:
                    self.summary_stats = pickle.load(f)

            # This is just to pull all the finished stuff in one place to make
            # post-processing easier
            finished_filename = Path(
                f"{str(Path(*self.result_path.parts[-3:])).replace('/', 'X')}.p"
            )
            self.finished_path = Path(
                self.result_path.parents[2], "finished", finished_filename
            )
            if not self.finished_path.parent.exists():
                self.finished_path.parent.mkdir()
            finish_params = copy.deepcopy(self.params)
            finish_params["best_precision"] = self.best_precision
            finish_params["universe"] = self.result_path.parts[-3]
            finish_params["result_path"] = self.result_path

            with open(self.finished_path, "wb") as f:
                pickle.dump(finish_params, f)

            self.schedule_plots(pdet)

    def create_schedule(self, pdet, universe_dir, workers):
        schedule_path = Path(
            universe_dir,
            "imaging_schedules",
            f"schedule_{self.hash}.p".replace(" ", ""),
        )
        if schedule_path.exists():
            logger.info(f"Loading schedule from {schedule_path}")
            with open(schedule_path, "rb") as f:
                final_df = pickle.load(f)
        else:
            # Make path
            schedule_path.parent.mkdir(exist_ok=True)

            SS = pdet.SS
            self.start_time = SS.TimeKeeping.missionStart
            self.start_time_jd = self.start_time.jd
            self.end_time = self.start_time + self.sim_length
            self.end_time_jd = self.end_time.jd
            self.obs_times = Time(
                np.arange(
                    self.start_time_jd,
                    self.end_time_jd,
                    self.block_length.to(u.d).value,
                ),
                format="jd",
                scale="tai",
            )
            self.obs_times_datetime = self.obs_times.datetime
            self.obs_times_jd = self.obs_times.jd
            self.n_obs_times = len(self.obs_times)
            self.block_inds = np.arange(0, self.n_obs_times, 1)

            self.min_int_time = self.block_length.to(u.d).value
            self.int_times_blocks = sorted(self.block_multiples)
            self.int_times = np.array(self.block_multiples) * self.block_length.to(u.d)
            self.int_times_d = self.int_times.to(u.d).value
            self.n_int_times = len(self.int_times)

            self.min_wait_blocks = math.ceil(
                self.min_required_wait_time.to(u.d).value / self.min_int_time
            )
            self.max_wait_blocks = math.ceil(
                self.max_required_wait_time.to(u.d).value / self.min_int_time
            )

            mode = list(
                filter(
                    lambda mode: mode["detectionMode"],
                    SS.TargetList.OpticalSystem.observingModes,
                )
            )[0]
            self.all_coeffs = {}
            self.koTimes = SS.koTimes
            self.koTimes_jd = self.koTimes.jd
            self.koMaps = SS.koMaps[mode["syst"]["name"]]
            self.all_pdets = pdet.pdets
            self.relevant_stars = list(self.all_pdets.keys())

            # Getting overhead time
            OS = SS.TargetList.OpticalSystem
            mode = list(
                filter(lambda mode: mode["detectionMode"] is True, OS.observingModes)
            )[0]
            self.ohtime = SS.Observatory.settlingTime + mode["syst"]["ohTime"]
            self.ohblocks = math.ceil(
                (self.ohtime / self.block_length).decompose().value
            )

            # Setting up solver
            self.star_planets = {}
            for star in self.relevant_stars:
                star_xr = self.all_pdets[star].pdet
                self.star_planets[star] = star_xr.planet.values
            model = cp_model.CpModel()
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = workers
            solver.parameters.log_search_progress = self.log_search_progress
            solver.parameters.max_time_in_seconds = self.max_time_in_seconds
            # Coefficients (probability of detections)

            # Input sanity checks
            max_schedule_int_time = max(self.int_times_d)
            max_pdet_int_time = max(
                self.all_pdets[self.relevant_stars[0]].int_time.values
            )
            assert max_schedule_int_time <= max_pdet_int_time, (
                "Probability of detection cannot be interpolated safely."
                "The highest integration time used to calculate probability of"
                f" detection ({max_pdet_int_time}) must be less than "
                "or equal to and the highest integration time expected to be used for"
                f" scheduling ({max_schedule_int_time})."
            )
            min_schedule_int_time = min(self.int_times_d)
            min_pdet_int_time = min(
                self.all_pdets[self.relevant_stars[0]].int_time.values
            )
            assert min_schedule_int_time >= min_pdet_int_time, (
                "Probability of detection cannot be interpolated safely."
                "The lowest integration time used to calculate probability of"
                f" detection ({min_pdet_int_time}) must be greater than "
                "or equal to and the lowest integration time expected to be used for"
                f" scheduling ({min_schedule_int_time})."
            )
            #########################################
            # Stage 1 - Detections above threshold
            #########################################
            logger.info("Creating decision variables and coefficients for optimization")
            model, vars = self.create_variables_and_coefficents(model, pdet)

            logger.info("Creating optimization constraints")
            # model.AddNoOverlap(vars["all_intervals"])
            model = self.same_star_observation_wait_constraint(model, vars, pdet)
            model = self.overhead_time_constraint(model, vars)

            logger.info("Setting up objective function")
            model, vars = self.create_observation_booleans(model, vars)
            model = self.create_objective_function(model, vars)
            logger.info("Running optimization solver")
            solver.Solve(model)
            final_df = self.process_result(vars, solver, pdet)

            with open(schedule_path, "wb") as f:
                pickle.dump(final_df, f)
        self.schedule = final_df

    def create_variables_and_coefficents(
        self, model, pdet, fixed_observations={}, prev_solver=None
    ):
        vars = {}
        vars["all_intervals"] = []

        for star in tqdm(self.relevant_stars, desc="stars", position=0):
            # star_name = star.replace("_", " ")
            star_ind = np.where(pdet.SS.TargetList.Name == star)[0][0]
            star_xr = pdet.pdets[star].pdet
            self.star_planets[star] = star_xr.planet.values
            # Array of booleans for each observation window, for current star
            # True means the star is observable
            obs_window_ko = np.array(
                np.floor(
                    np.interp(self.obs_times_jd, self.koTimes_jd, self.koMaps[star_ind])
                ),
                dtype=bool,
            )
            vars[star] = {"vars": []}
            if star in fixed_observations.keys():
                star_fixed_observations = fixed_observations[star]
            else:
                star_fixed_observations = []
            star_intervals = min(
                self.max_observations_per_star,
                self.requested_planet_observations * len(self.star_planets[star]),
            )
            n_fixed_observations = len(star_fixed_observations)
            n_variable_observations = star_intervals - n_fixed_observations
            for i in range(star_intervals):
                if i < n_fixed_observations:
                    # Fixed intervals
                    _startval = prev_solver.Value(star_fixed_observations[i]["start"])
                    _start = model.NewIntVar(
                        _startval, _startval, f"{star} start {i}, fixed"
                    )
                    _sizeval = prev_solver.Value(star_fixed_observations[i]["size"])
                    _size = model.NewIntVar(
                        _sizeval, _sizeval, f"{star} size {i}, fixed"
                    )
                    _endval = prev_solver.Value(star_fixed_observations[i]["end"])
                    _end = model.NewIntVar(_endval, _endval, f"{star} end {i}, fixed")
                    _active = model.NewIntVar(1, 1, f"{star} active {i}, fixed")
                    _interval = model.NewIntervalVar(
                        _start, _size, _end, f"{star} interval {i}, fixed"
                    )
                else:
                    # interval_inds = np.where(star_observing_bins[star] == i)[0]
                    start_domain = self.block_inds[i::n_variable_observations]
                    end_domain = []
                    for block_size in self.int_times_blocks:
                        end_domain.extend((start_domain + block_size).tolist())
                    end_domain = np.unique(end_domain)
                    end_domain = end_domain[end_domain < self.n_obs_times]
                    # Creating intervals that can be moved
                    _start = model.NewIntVarFromDomain(
                        cp_model.Domain.FromValues(start_domain),
                        f"{star} start {i}",
                    )
                    _size = model.NewIntVarFromDomain(
                        cp_model.Domain.FromValues(self.int_times_blocks),
                        f"{star} size {i}",
                    )
                    _end = model.NewIntVarFromDomain(
                        cp_model.Domain.FromValues(end_domain),
                        f"{star} end {i}",
                    )
                    _active = model.NewBoolVar(f"{star} active {i}")
                    _interval = model.NewOptionalIntervalVar(
                        _start, _size, _end, _active, f"{star} interval {i}"
                    )
                vars["all_intervals"].append(_interval)
                vars[star]["vars"].append(
                    {
                        "star": star,
                        "start": _start,
                        "size": _size,
                        "end": _end,
                        "active": _active,
                        "interval": _interval,
                        "start_domain": start_domain,
                    }
                )

            # Create the coefficent array, for each observation var we have
            # pdet values for each planet around a star
            star_pdets = np.zeros(
                (
                    len(self.star_planets[star]),
                    self.n_int_times,
                    self.n_obs_times,
                )
            )
            for planet in self.star_planets[star]:
                # shape is (len(int_times), len(obs_times))
                planet_pdet = np.array(
                    pdet.pdets[star]
                    .sel(planet=planet)
                    .interp(time=self.obs_times_datetime, int_time=self.int_times_d)
                    .pdet
                )

                # CPSat solver only works with integers, cast pdet values to
                # integers by multiplying by 100 and cutting
                int_planet_pdet = np.array(self.coeff_multiple * planet_pdet, dtype=int)
                # Loop through and set any observations that would go into
                # keepout to 0
                for obs_ind, _ in enumerate(self.obs_times_jd):
                    for int_ind, int_time_d in enumerate(self.int_times_d):
                        # Number of observing windows this observation spans
                        n_windows = int(int_time_d / self.min_int_time)

                        # Check that the observation doesn't intersect with keepout
                        ko_vals = obs_window_ko[obs_ind : obs_ind + n_windows]
                        # True if any ko_vals are False
                        in_keepout = ~np.all(ko_vals)
                        if in_keepout:
                            int_planet_pdet[int_ind][obs_ind] = 0
                star_pdets[planet][:][:] = int_planet_pdet
            vars[star]["coeffs"] = star_pdets
        return model, vars

    def same_star_observation_wait_constraint(self, model, vars, pdet):
        # Constraint on blocks between observations
        # This is done by looping over all the intervals for each star
        # creating an intermediary boolean that is true when they are both
        # active and adding a constraint that the intervals is above the user
        # defined distance between observations
        for star in self.relevant_stars:
            star_var_list = vars[star]["vars"]

            # Get the wait time between observations
            blocks_between_star_obs = int(len(self.obs_times) / 4)
            min_period = np.inf
            for planet in self.star_planets[star]:
                est_planet_period = np.median(pdet.pops[star][planet].T.to(u.d).value)
                if est_planet_period < min_period:
                    min_period = est_planet_period
            quarter_min_period_blocks = math.ceil(
                (est_planet_period / 4) / self.block_length.to(u.d).value
            )
            # if est_planet_blocks < blocks_between_star_obs:
            #     blocks_between_star_obs = est_planet_blocks
            # Goal of this constraint is to spread the observations out over a
            # reasonable time frame, for a short period planet we don't want to
            # wait too long between observations, and if the only planet in a
            # system has a very long period compared to the available
            # observation time then we still want to have multiple observations
            # of it so we have a min wait time and a max wait time, if it's between
            # those values then we set it to a quarter of the shortest period
            # in the system,
            if quarter_min_period_blocks < self.min_wait_blocks:
                blocks_between_star_obs = self.min_wait_blocks
            elif quarter_min_period_blocks > self.max_wait_blocks:
                blocks_between_star_obs = self.max_wait_blocks
            else:
                blocks_between_star_obs = quarter_min_period_blocks

            for i, i_var_set in enumerate(star_var_list[:-1]):
                # Get stage_1_model variables
                i_start = i_var_set["start"]
                i_end = i_var_set["end"]
                i_var_set["active"]

                for j, j_var_set in enumerate(star_var_list[i + 1 :]):
                    # Get stage_1_model variables
                    j_start = j_var_set["start"]
                    j_end = j_var_set["end"]
                    j_var_set["active"]

                    # Bool used to determine which distance we need to be checking
                    i_before_j_bool = model.NewBoolVar(
                        f"{star} interval {i} starts before interval {j}"
                    )
                    model.Add(i_start < j_start).OnlyEnforceIf(i_before_j_bool)
                    model.Add(i_start > j_start).OnlyEnforceIf(i_before_j_bool.Not())

                    # Add distance constraints
                    model.Add(j_start - i_end >= blocks_between_star_obs).OnlyEnforceIf(
                        i_before_j_bool
                    )
                    model.Add(i_start - j_end >= blocks_between_star_obs).OnlyEnforceIf(
                        i_before_j_bool.Not()
                    )
        return model

    def overhead_time_constraint(self, model, vars):
        # Constraint that makes sure that observations account for overhead time
        # This is done by looping over all the intervals for each star
        # creating an intermediary boolean that is true when they are both
        # active and adding a constraint that the intervals is above the user
        # defined distance between observations
        all_interval_dicts = []
        for star in self.relevant_stars:
            all_interval_dicts.extend(vars[star]["vars"])
        interval_combinations = itertools.combinations(all_interval_dicts, 2)
        for int1, int2 in interval_combinations:
            int1_star = int1["star"]
            int1_start = int1["start"]
            int1_end = int1["end"]
            int1["active"]
            int1_interval = int1["interval"]

            int2_star = int2["star"]
            int2_start = int2["start"]
            int2_end = int2["end"]
            int2["active"]
            int2_interval = int2["interval"]
            if int1_star == int2_star:
                continue

            # Bool to determine which distance we need to be checking
            int1_before_int2_bool = model.NewBoolVar(
                f"{int1_interval} before {int2_interval}"
            )
            model.Add(int1_start < int2_start).OnlyEnforceIf(int1_before_int2_bool)
            model.Add(int1_start > int2_start).OnlyEnforceIf(
                int1_before_int2_bool.Not()
            )

            # Add distance constraints
            model.Add(int2_start - int1_end >= self.ohblocks).OnlyEnforceIf(
                int1_before_int2_bool
            )
            model.Add(int1_start - int2_end >= self.ohblocks).OnlyEnforceIf(
                int1_before_int2_bool.Not()
            )
        return model

    def create_observation_booleans(self, model, vars):
        vars["obj_terms"] = []
        vars["active_bools"] = []
        vars["size_vars"] = []
        vars["planet_terms"] = {}
        for star in tqdm(self.relevant_stars, desc="Creating observation booleans"):
            # list of the dictionary of interval variables
            star_var_list = vars[star]["vars"]

            # Pdet values for this star
            coeffs = vars[star]["coeffs"]

            # Dictionary keyed on [star][planet][interval_n] that gets used to
            # create the constraint for the planets
            vars["planet_terms"][star] = {}
            n_intervals = min(
                self.max_observations_per_star,
                self.requested_planet_observations * len(self.star_planets[star]),
            )
            for interval_n in range(n_intervals):
                vars["planet_terms"][star][interval_n] = {}
                for planet in self.star_planets[star]:
                    vars["planet_terms"][star][interval_n][planet] = []

            # Loop over all the intervals, defined by start, size, end,
            # active, for this star
            for i, var_set in enumerate(star_var_list):
                start_var = var_set["start"]
                size_var = var_set["size"]
                active_var = var_set["active"]
                start_domain = var_set["start_domain"]
                vars["active_bools"].append(active_var)
                vars["size_vars"].append(size_var)
                for n_obs in start_domain:
                    prev_above_threshold = 0
                    for n_int, n_times_blocks in enumerate(self.int_times_blocks):
                        if (n_obs + n_times_blocks) >= self.n_obs_times:
                            # Cannot end after the last obs time
                            continue
                        current_above_threshold = sum(
                            coeffs[:, n_int, n_obs]
                            >= self.coeff_multiple * self.planet_threshold
                        )
                        if current_above_threshold <= prev_above_threshold:
                            # No benefit to this stage
                            continue
                        _bool = model.NewBoolVar(f"{star} {n_int} {n_obs}")
                        model.Add(start_var == n_obs).OnlyEnforceIf(_bool)
                        model.Add(size_var == n_times_blocks).OnlyEnforceIf(_bool)
                        model.Add(active_var == 1).OnlyEnforceIf(_bool)
                        vars["obj_terms"].append(_bool * sum(coeffs[:, n_int, n_obs]))
                        for planet in self.star_planets[star]:
                            vars["planet_terms"][star][i][planet].append(
                                (_bool, coeffs[planet, n_int, n_obs], n_obs)
                            )
                        prev_above_threshold = current_above_threshold
        return model, vars

    def create_objective_function(self, model, vars):
        # Maximize the number of observations above the threshold, with a cap
        # of benefit set by the requested_planet_observations parameter
        all_planet_requested_observations = []
        amounts_above_threshold_terms = []
        for star in tqdm(self.relevant_stars, desc="Adding threshold constraint"):
            n_intervals = min(
                self.max_observations_per_star,
                self.requested_planet_observations * len(self.star_planets[star]),
            )
            for planet in self.star_planets[star]:
                above_requested_observations = model.NewBoolVar(
                    f"{star}{planet} above requested observations"
                )
                requested_observations = model.NewIntVar(
                    0,
                    self.requested_planet_observations,
                    f"{star}{planet} meets threshold",
                )
                above_threshold_val = []
                for interval_n in range(n_intervals):
                    for _bool, coeff, _ in vars["planet_terms"][star][interval_n][
                        planet
                    ]:
                        if coeff >= self.coeff_multiple * self.planet_threshold:
                            amounts_above_threshold_terms.append(
                                _bool
                                * (coeff - self.coeff_multiple * self.planet_threshold)
                            )
                            above_threshold_val.append(_bool)

                # Defining the above_requested_observations boolean
                model.Add(
                    sum(above_threshold_val) > self.requested_planet_observations
                ).OnlyEnforceIf(above_requested_observations)
                model.Add(
                    sum(above_threshold_val) <= self.requested_planet_observations
                ).OnlyEnforceIf(above_requested_observations.Not())

                # Changing the value of requested_observations based on the
                # above_requested_observations boolean
                model.Add(
                    requested_observations == self.requested_planet_observations
                ).OnlyEnforceIf(above_requested_observations)
                model.Add(
                    requested_observations == sum(above_threshold_val)
                ).OnlyEnforceIf(above_requested_observations.Not())
                all_planet_requested_observations.append(requested_observations)
        model.Maximize(
            sum(
                self.coeff_multiple
                * max(self.block_multiples)
                * all_planet_requested_observations
            )
            + sum(amounts_above_threshold_terms)
            - sum(vars["size_vars"])
            - sum(vars["active_bools"])
        )
        return model

    def process_result(self, vars, solver, pdet):
        planet_ub = max([len(item) for _, item in self.star_planets.items()])
        final_sInds = []
        final_stars = []
        final_times = []
        final_times_jd = []
        final_int_times = []
        final_int_times_d = []
        total_coeffs = []
        planet_thresholds = {}
        planet_coeffs = {}
        planet_periods = {}
        planet_as = {}
        for pnum in range(planet_ub):
            planet_coeffs[pnum] = []
            planet_thresholds[pnum] = []
            planet_periods[pnum] = []
            planet_as[pnum] = []
        coeffs_lists = []
        thresholds_lists = []
        periods_lists = []
        as_lists = []  # Semi-major axis
        pop_lists = []
        WAs_lists = []
        dMags_lists = []
        # Create dataframe of the observations
        for star in self.relevant_stars:
            star_var_list = vars[star]["vars"]
            coeffs = vars[star]["coeffs"]
            for var_set in star_var_list:
                start_var = var_set["start"]
                size_var = var_set["size"]
                active_var = var_set["active"]
                if solver.Value(active_var):
                    n_obs = solver.Value(start_var)
                    n_int = solver.Value(size_var)
                    n_int_block = np.where(n_int == np.array(self.int_times_blocks))[0][
                        0
                    ]
                    final_sInds.append(np.where(pdet.SS.TargetList.Name == star)[0][0])
                    final_stars.append(star)
                    final_times.append(self.obs_times[n_obs])
                    final_times_jd.append(self.obs_times_jd[n_obs])
                    final_int_times.append(self.int_times[n_int_block])
                    final_int_times_d.append(self.int_times[n_int_block].to(u.d).value)
                    above_threshold = False
                    summed_coeff = 0
                    coeffs_list = []
                    thresholds_list = []
                    periods_list = []
                    as_list = []
                    WAs_list = []
                    dMags_list = []
                    for pnum in range(planet_ub):
                        if pnum in self.star_planets[star]:
                            coeff = coeffs[pnum, n_int_block, n_obs]
                            summed_coeff += coeff
                            above_threshold = (
                                coeff >= self.coeff_multiple * self.planet_threshold
                            )
                            period = np.median(pdet.pops[star][pnum].T.to(u.yr).value)
                            semi_major_axis = np.median(
                                pdet.pops[star][pnum].a.to(u.AU).value
                            )
                            coeffs_list.append(coeff)
                            thresholds_list.append(above_threshold)
                            periods_list.append(period)
                            as_list.append(semi_major_axis)
                            WA, dMag = pdet.pops[star][pnum].prop_for_imaging(
                                self.obs_times[n_obs]
                            )
                            WAs_list.append(np.median(WA.to(u.arcsec).value))
                            dMags_list.append(np.median(dMag))
                        else:
                            coeff = np.nan
                            above_threshold = np.nan
                            period = np.nan
                            semi_major_axis = np.nan
                        planet_coeffs[pnum].append(coeff)
                        planet_thresholds[pnum].append(above_threshold)
                        planet_periods[pnum].append(period)
                        planet_as[pnum].append(semi_major_axis)
                    coeffs_lists.append(coeffs_list)
                    thresholds_lists.append(thresholds_list)
                    periods_lists.append(periods_list)
                    as_lists.append(as_list)
                    pop_lists.append(pdet.pops[star])
                    WAs_lists.append(WAs_list)
                    dMags_lists.append(dMags_list)
                    total_coeffs.append(summed_coeff)

        final_df = pd.DataFrame()
        final_df["sInd"] = final_sInds
        final_df["star"] = final_stars
        final_df["time"] = final_times
        final_df["time_jd"] = final_times_jd
        final_df["int_time"] = final_int_times
        final_df["int_time_d"] = final_int_times_d
        final_df["total_coeffs"] = total_coeffs
        threshold_cols = []
        for pnum in range(planet_ub):
            final_df[f"coeff{pnum}"] = planet_coeffs[pnum]
        for pnum in range(planet_ub):
            threshold_col = f"threshold{pnum}"
            threshold_cols.append(threshold_col)
            final_df[threshold_col] = planet_thresholds[pnum]
        for pnum in range(planet_ub):
            final_df[f"expected_period{pnum}"] = planet_periods[pnum]
        for pnum in range(planet_ub):
            final_df[f"expected_a{pnum}"] = planet_as[pnum]
        final_df["expected_detections"] = (
            final_df.fillna(0)[threshold_cols].astype("int").sum(axis=1)
        )
        final_df["all_planet_coeffs"] = coeffs_lists
        final_df["all_planet_thresholds"] = thresholds_lists
        final_df["all_planet_periods"] = periods_lists
        final_df["all_planet_as"] = as_lists
        final_df["all_planet_pops"] = pop_lists
        final_df["dMags"] = dMags_lists
        final_df["WAs"] = WAs_lists
        return final_df

    def schedule_plots(self, pdet):
        SS = pdet.SS
        SU = SS.SimulatedUniverse
        start_time_jd = SS.TimeKeeping.missionStart.jd
        end_time_jd = start_time_jd + self.sim_length.to(u.d).value
        obs_times = Time(
            np.arange(start_time_jd, end_time_jd, self.block_length.to(u.d).value),
            format="jd",
            scale="tai",
        )
        figsc, axsc = plt.subplots(figsize=(10, 7))

        # Could do colorbars like the SPIE paper for the 1 day pdet values
        nplans = len(SU.plan2star)
        star_names = SS.TargetList.Name[SU.plan2star]
        star_counter = Counter(star_names)
        planet_names = {}
        pInd = 0
        used_stars = []
        for star_name in star_names:
            if star_name in used_stars:
                continue
            else:
                used_stars.append(star_name)
            for pnum in range(star_counter[star_name]):
                planet_names[pInd] = f"{star_name}{chr(ord('a')+ pnum)}"
                pInd += 1

        # Planets get put at y=pInd+1
        axsc.set_ylim([0, nplans + 2])
        axsc.set_yticks(np.arange(0, nplans + 2, 1))
        tick_labels = [""]
        for pind, label in enumerate(planet_names.values()):
            if pind in self.targetdf.columns:
                success = self.targetdf[pind]["success"]
                fail = self.targetdf[pind]["fail"]
                tick_labels.append(f"{success}/{success+fail}-{label}")
            else:
                tick_labels.append(f"0/0 - {label}")

        # tick_labels = list(planet_names.values())
        # tick_labels.insert(0, "")
        tick_labels.append("")
        axsc.set_yticklabels(tick_labels)
        axsc.set_xlim([0, end_time_jd - start_time_jd])
        axsc.set_xlabel("Time since mission start (d)")

        axsc.set_title(
            f"Observing schedule, "
            f"RV sigma: {self.best_precision} m/s, "
            f"fEZ_q:{pdet.fEZ_quantile:.2f}"
        )
        for yval in range(nplans + 2):
            axsc.axhline(y=yval + 0.5, alpha=0.5, ls="--")

        pdet_int_time = (
            np.median(self.block_multiples) * self.block_length.to(u.d).value
        )
        cmap = plt.get_cmap("viridis")
        # pdet_cbar_ax = figsc.add_axes()
        cbar_2 = figsc.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap), ax=axsc, alpha=0.5, location="right"
        )
        cbar_2.set_label(r"$P_{det}(t_{int}=$" + f"{pdet_int_time:.0f}d)")

        det_colors = {0: "red", 1: "lime", -1: "yellow", -2: "yellow"}
        det_colors_schedule = copy.deepcopy(det_colors)
        det_colors_schedule[1] = "black"
        n_inds = 50
        plot_times = obs_times[::10]
        dt = 10 * self.block_length.to(u.d).value
        SS.reset_sim(genNewPlanets=False)
        used_sInds = []
        for system_name in tqdm(pdet.pops.keys(), desc="Generating plots"):
            system_pdets = pdet.pdets[system_name]

            pops = pdet.pops[system_name]
            for pval, pop in enumerate(pops):
                planet_pdet = system_pdets.pdet[pval]
                pdet_vals = planet_pdet.interp(
                    time=obs_times.datetime, int_time=pdet_int_time
                ).values

                # Have to find the right pInd to plot on since it's not
                # in the same order
                pop = pdet.pops[system_name][pval]
                sInd = np.where(SS.TargetList.Name == system_name)[0][0]
                pInds = np.where(SU.plan2star == sInd)[0]
                pInd = pInds[np.argmin(np.abs(np.median(pop.a) - SU.a[pInds]))]
                extent = 0, end_time_jd - start_time_jd, pInd + 0.5, pInd + 1.5
                axsc.imshow(
                    np.expand_dims(pdet_vals, axis=1).T,
                    aspect="auto",
                    interpolation="none",
                    extent=extent,
                    cmap=cmap,
                    norm=mpl.colors.Normalize(0, 1),
                    alpha=0.5,
                    zorder=0,
                )
                if system_name not in self.schedule.star.values:
                    continue
                fig, (ax_WA, ax_dMag) = plt.subplots(figsize=(10, 5), ncols=2)
                # sInd = np.where(SS.TargetList.Name == system_name)[0][0]
                # pInds = np.where(SU.plan2star == sInd)[0]
                # pInd = pInds[np.argmin(np.abs(np.median(pop.a) - SU.a[pInds]))]
                if pInd not in self.targetdf.columns:
                    continue
                # if pInd == 22:
                #     breakpoint()
                if sInd in used_sInds:
                    # If this isn't here then things get messed up for
                    # multiple-planet systems fitted and observed
                    SS.reset_sim(genNewPlanets=False)
                else:
                    used_sInds.append(sInd)

                fig_path = Path(
                    f"{self.result_path}/{system_name.replace(' ', '_')}_"
                    f"{pInd}_{self.best_precision}.png"
                )

                # if fig_path.exists():
                #     continue
                eWAs = []
                edMags = []
                pdMags = np.zeros((len(plot_times), n_inds))
                pWAs = np.zeros((len(plot_times), n_inds))
                for i, obs_time in enumerate(plot_times):
                    SU.propag_system(sInd, dt * u.d)
                    edMags.append(SU.dMag[pInd])
                    eWAs.append(SU.WA[pInd].to(u.arcsec).value)
                    pWA, pdMag = pop.prop_for_imaging(obs_time)
                    pdMags[i] = pdMag[:n_inds]
                    pWAs[i] = pWA[:n_inds].to(u.arcsec).value
                colors = cmap(np.linspace(0, 1, n_inds))
                for j in range(n_inds):
                    ax_dMag.plot(
                        plot_times.jd - plot_times[0].jd,
                        pdMags[:, j],
                        color=colors[j],
                        label=f"{j}",
                        alpha=0.25,
                    )
                    ax_WA.plot(
                        plot_times.jd - plot_times[0].jd,
                        pWAs[:, j],
                        color=colors[j],
                        label=f"{j}",
                        alpha=0.25,
                    )
                ax_dMag.plot(plot_times.jd - plot_times[0].jd, edMags, color="k")
                ax_WA.plot(plot_times.jd - plot_times[0].jd, eWAs, color="k")

                ax_dMag.set_ylim([15, 35])
                ax_WA.set_ylim([0, 0.3])
                dMagLims = ax_dMag.get_ylim()
                dMagheight = dMagLims[1] - dMagLims[0]
                WALims = ax_WA.get_ylim()
                WAheight = WALims[1] - WALims[0]
                intstr = ""
                SNRstr = ""
                for nobs, _t in enumerate(self.targetdf[pInd].obs_time):
                    _det_status = self.targetdf[pInd].det_status[nobs]
                    _tint = self.targetdf[pInd].int_time[nobs].to(u.d).value
                    zeroed_time = _t.jd - plot_times[0].jd
                    dMag_sq = mpl.patches.Rectangle(
                        (zeroed_time, dMagLims[0]),
                        width=_tint,
                        height=dMagheight,
                        zorder=0,
                        color=det_colors[_det_status],
                    )
                    ax_dMag.add_patch(dMag_sq)
                    WA_sq = mpl.patches.Rectangle(
                        (zeroed_time, WALims[0]),
                        width=_tint,
                        height=WAheight,
                        zorder=0,
                        color=det_colors[_det_status],
                    )
                    ax_WA.add_patch(WA_sq)
                    intstr += f"{_tint:.2f}, "
                    SNRstr += f"{self.targetdf[pInd]['SNR'][nobs]:.2f}, "

                    obs_sq = mpl.patches.Rectangle(
                        (zeroed_time, pInd + 0.5),
                        width=_tint,
                        height=1,
                        zorder=1,
                        # color="white",
                        edgecolor=det_colors_schedule[_det_status],
                        hatch=r"\\",
                        alpha=1,
                        fill=False,
                    )
                    axsc.add_patch(obs_sq)
                    axsc.axvline(x=zeroed_time, alpha=0.25)
                    axsc.axvline(x=zeroed_time + _tint, alpha=0.25, ls="--")

                fEZstr = f"{self.targetdf[pInd]['fEZ'][nobs].value:.0e}"
                intstr = intstr[:-2]
                SNRstr = SNRstr[:-2]

                ax_dMag.set_ylabel(r"$\Delta$mag")
                ax_WA.set_ylabel('Planet-star angular separation (")')
                ax_WA.set_xlabel("Time since mission start (d)")
                ax_dMag.set_xlabel("Time since mission start (d)")
                fig.suptitle(
                    f"{system_name}, "
                    f"RV precision: {self.best_precision} m/s, "
                    f"int time: [{intstr}] d, "
                    f"SNRs: [{SNRstr}], "
                    f"fEZ: {fEZstr}"
                )
                fig.savefig(fig_path, dpi=300)
        figsc_path1 = Path(f"{self.result_path}/full_schedule.png")
        figsc_path2 = Path(f"{self.finished_path}ng")
        figsc.tight_layout()
        figsc.savefig(figsc_path1, dpi=300)
        figsc.savefig(figsc_path2, dpi=300)
