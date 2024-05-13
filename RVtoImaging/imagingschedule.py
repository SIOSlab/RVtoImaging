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
        necessary_param_names = [
            "coeff_multiple",
            "sim_length",
            "block_length",
            "block_multiples",
            "max_observations_per_star",
            "planet_threshold",
            "requested_planet_observations",
            "min_required_wait_time",
            "max_required_wait_time",
            "max_time_in_seconds",
        ]
        self.params = params
        necessary_params = {"pdet_hash": pdet.settings_hash}
        for _param in necessary_param_names:
            assert (
                _param in params.keys()
            ), f"Scheduler parameters does not have {_param} parameter"
            setattr(self, _param, params[_param])
            necessary_params[_param] = params[_param]
        # self.coeff_multiple = params["coeff_multiple"]
        # self.sim_length = params["sim_length"]
        # self.block_length = params["block_length"]
        # self.block_multiples = params["block_multiples"]
        # self.max_observations_per_star = params["max_observations_per_star"]
        # self.planet_threshold = params["planet_threshold"]
        # self.requested_planet_observations = params["requested_planet_observations"]
        # self.min_required_wait_time = params["min_required_wait_time"]
        # self.max_required_wait_time = params["max_required_wait_time"]
        # self.max_time_in_seconds = params["max_time_in_seconds"]

        opt_params = params.get("opt")
        opt_param_defaults = {
            "log_search_progress": True,
            "random_walk_method": "25dMag",
            "n_random_walks": 300,
            "schedule_hint": None,
        }
        if opt_params is not None:
            for _param, _val in opt_param_defaults.items():
                if _param in opt_params.keys():
                    setattr(self, _param, opt_params[_param])
                else:
                    setattr(self, _param, _val)
                    logger.info(
                        (
                            "Setting optional scheduler parameter "
                            f"{_param} to default of {_val}"
                        )
                    )
        else:
            # Default values
            for _param, _val in opt_param_defaults.items():
                setattr(self, _param, _val)
                logger.info(
                    (
                        "Setting optional scheduler parameter "
                        f"{_param} to default of {_val}"
                    )
                )
        # Add the keeoput angles to the hash
        mode = list(
            filter(
                lambda mode: mode["detectionMode"],
                pdet.SS.TargetList.OpticalSystem.observingModes,
            )
        )[0]
        syst = mode["syst"]
        if sun_ko := syst.get("koAngles_Sun"):
            necessary_params["koAngles_Sun"] = sun_ko
        if earth_ko := syst.get("koAngles_Earth"):
            necessary_params["koAngles_Earth"] = earth_ko
        if moon_ko := syst.get("koAngles_Moon"):
            necessary_params["koAngles_Moon"] = moon_ko
        if small_ko := syst.get("koAngles_Small"):
            necessary_params["koAngles_Small"] = small_ko

        ################

        # Create a hash to identify this schedule
        self.hash = genHexStr(dictToSortedStr(necessary_params))

        self.result_path = Path(universe_dir, "results", f"schedule_{self.hash}")
        self.result_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Creating Imaging Schedule {self.hash}"
            f" for {universe_dir.parts[-1].replace('_', ' ')} "
        )
        self.create_schedule(pdet, universe_dir, workers)

        intervals = pd.arrays.IntervalArray.from_tuples(
            tuple(
                zip(
                    self.schedule.time_jd.values,
                    (self.schedule.time_jd + self.schedule.int_time_d).values,
                )
            )
        )
        if np.any(
            [
                np.sum(intervals.overlaps(intervals[i])) - 1
                for i in range(len(intervals))
            ]
        ):
            logger.warn("Schedule did not respect overlap constraint")
            breakpoint()
        self.total_success = 0
        self.total_obs = 0
        self.total_int_time = 0
        self.total_planets = len(pdet.SS.SimulatedUniverse.plan2star)
        self.planets_fitted = 0
        self.unique_planets_detected = 0
        self.per_planet_data = []
        # self.per_planet_completeness = pdet.SS.Completeness.int_comp

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
        finish_params["fEZ_quantile"] = pdet.fEZ_quantile
        finish_params["universe"] = self.result_path.parts[-3]
        finish_params["result_path"] = self.result_path
        # Add schedule to the SS module
        if self.schedule.empty:
            logger.warn(f"No observations scheduled for {self.result_path}")
            finish_params["schedule_found"] = False
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
                self.flatdf = self.add_percentages(self.flatdf, pdet)

                self.summary_stats["Sun_ko"] = (
                    syst.get("koAngles_Sun").to(u.deg).value.tolist()
                )
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
            int_times = self.block_multiples * self.block_length
            random_walk_path = Path(
                self.result_path,
                f"random_walk_{self.random_walk_method}_{self.n_random_walks}.p",
            )
            if random_walk_path.exists():
                with open(random_walk_path, "rb") as f:
                    self.random_walk_res = pickle.load(f)
            else:
                if self.random_walk_method == "25dMag":
                    self.random_walk_res = pdet.SS.random_walk_sim(
                        self.n_random_walks, self.sim_length
                    )
                else:
                    self.random_walk_res = pdet.SS.random_walk_sim(
                        self.n_random_walks, self.sim_length, int_times=int_times
                    )
                with open(random_walk_path, "wb") as f:
                    pickle.dump(self.random_walk_res, f)

            # self.schedule_plots(pdet)
            finish_params["schedule_found"] = True

        finish_params["total_detections"] = self.total_success
        finish_params["total_observations"] = self.total_obs
        finish_params["total_int_time"] = self.total_int_time
        finish_params["total_planets"] = self.total_planets
        finish_params["planets_fitted"] = self.planets_fitted
        finish_params["unique_planets_detected"] = self.unique_planets_detected
        finish_params["per_planet_data"] = self.per_planet_data

        with open(self.finished_path, "wb") as f:
            pickle.dump(finish_params, f)

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
            model, vars = self.same_star_observation_wait_constraint(model, vars, pdet)
            model, vars = self.overhead_time_constraint(model, vars)

            logger.info("Setting up objective function")
            model, vars = self.create_observation_booleans(model, vars)
            model, vars = self.create_objective_function(model, vars)

            if self.schedule_hint is not None:
                model = self.add_schedule_hints(model, vars, self.schedule_hint)

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
            # True means the star is in keepout
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
        vars["i_before_j_bools"] = {}
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
                i_interval = i_var_set["interval"]
                i_start = i_var_set["start"]
                i_end = i_var_set["end"]
                i_var_set["active"]

                for j, j_var_set in enumerate(star_var_list[i + 1 :]):
                    # Get stage_1_model variables
                    j_interval = j_var_set["interval"]
                    j_start = j_var_set["start"]
                    j_end = j_var_set["end"]
                    j_var_set["active"]

                    # Bool used to determine which distance we need to be checking
                    i_before_j_bool = model.NewBoolVar(
                        f"{star} interval {i} starts before interval {j}"
                    )
                    vars["i_before_j_bools"][(i_interval, j_interval)] = i_before_j_bool
                    model.Add(i_start < j_start).OnlyEnforceIf(i_before_j_bool)
                    model.Add(i_start > j_start).OnlyEnforceIf(i_before_j_bool.Not())

                    # Add distance constraints
                    model.Add(j_start - i_end >= blocks_between_star_obs).OnlyEnforceIf(
                        i_before_j_bool
                    )
                    model.Add(i_start - j_end >= blocks_between_star_obs).OnlyEnforceIf(
                        i_before_j_bool.Not()
                    )
        return model, vars

    def overhead_time_constraint(self, model, vars):
        # Constraint that makes sure that observations account for overhead time
        # This is done by looping over all the intervals for each star
        # creating an intermediary boolean that is true when they are both
        # active and adding a constraint that the intervals is above the user
        # defined distance between observations
        vars["int1_before_int2_bools"] = {}
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
            vars["int1_before_int2_bools"][(int1_interval, int2_interval)] = (
                int1_before_int2_bool,
                int1_start,
                int2_start,
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
        return model, vars

    def create_observation_booleans(self, model, vars):
        vars["obj_terms"] = []
        vars["active_bools"] = []
        vars["size_vars"] = []
        vars["end_vars"] = []
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
                vars["end_vars"].append(var_set["end"])
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
                                (
                                    _bool,
                                    coeffs[planet, n_int, n_obs],
                                    n_obs,
                                    n_times_blocks,
                                )
                            )
                        prev_above_threshold = current_above_threshold
        return model, vars

    def create_objective_function(self, model, vars):
        # Maximize the number of observations above the threshold, with a cap
        # of benefit set by the requested_planet_observations parameter
        all_planet_requested_observations = []
        amounts_above_threshold_terms = []
        all_planet_requested_observations_dict = {}
        above_requested_observations_dict = {}
        for star in tqdm(self.relevant_stars, desc="Adding threshold constraint"):
            n_intervals = min(
                self.max_observations_per_star,
                self.requested_planet_observations * len(self.star_planets[star]),
            )
            for planet in self.star_planets[star]:
                above_requested_observations = model.NewBoolVar(
                    f"{star}{planet} above requested observations"
                )
                above_requested_observations_dict[
                    (star, planet)
                ] = above_requested_observations
                requested_observations = model.NewIntVar(
                    0,
                    self.requested_planet_observations,
                    f"{star}{planet} meets threshold",
                )
                all_planet_requested_observations_dict[
                    (star, planet)
                ] = requested_observations
                above_threshold_val = []
                for interval_n in range(n_intervals):
                    for _bool, coeff, *_ in vars["planet_terms"][star][interval_n][
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
            # + sum(amounts_above_threshold_terms)
            # - sum(vars["size_vars"])
            - sum(vars["active_bools"])
            - sum(vars["end_vars"])
        )
        vars[
            "all_planet_requested_observations"
        ] = all_planet_requested_observations_dict
        vars["above_requested_observations"] = above_requested_observations_dict
        return model, vars

    def add_schedule_hints(self, model, vars, prev_schedule):
        logger.info("Adding solver hints from previous schedule")
        with open(prev_schedule, "rb") as f:
            prev_final_df = pickle.load(f)
        stars = prev_final_df["star"].unique()
        interval_start_hints = {}
        for star in stars:
            star_df = prev_final_df[prev_final_df["star"] == star]
            if star not in vars.keys():
                continue
            star_var_list = vars[star]["vars"]
            # Need to vary the start domains so that the intervals are not overlapping
            for i, var_set in enumerate(star_var_list):
                if star_df.empty:
                    continue
                interval_var = var_set["interval"]
                start_var = var_set["start"]
                size_var = var_set["size"]
                active_var = var_set["active"]
                start_domain = var_set["start_domain"]
                end_var = var_set["end"]

                domain_times = self.obs_times_jd[start_domain]
                obs_in_domain = [
                    time for time in star_df["time_jd"] if time in domain_times
                ]
                if len(obs_in_domain) > 0:
                    prev_obs = star_df[star_df["time_jd"] == obs_in_domain[0]]
                    n_obs = np.argwhere(obs_in_domain[0] == self.obs_times_jd)[0][0]
                    prev_obs = prev_obs.iloc[0]
                    prev_start = self.block_inds[n_obs]
                    prev_size = int(prev_obs["int_time_d"].item() / self.min_int_time)
                    prev_end = prev_start + prev_size
                    interval_start_hints[interval_var] = prev_start
                    model.AddHint(start_var, prev_start)
                    model.AddHint(size_var, prev_size)
                    model.AddHint(end_var, prev_end)
                    model.AddHint(active_var, 1)
                    # Go through all the booleans and add hints
                    for _bool, _, _n_obs, n_times_blocks in vars["planet_terms"][star][
                        i
                    ][0]:
                        correct_bool = (_n_obs == n_obs) & (prev_size == n_times_blocks)
                        model.AddHint(_bool, correct_bool)
                    # Create the star_df to not have this observation
                    star_df = star_df[star_df["time_jd"] != self.obs_times_jd[n_obs]]
                    logger.debug(f"Added hint for observation {n_obs} for star {star}")
                # This might be useful for some models but it functions well
                # enough without it
                # if not hints_added:
                #     start_val = default_starts[i]
                #     interval_start_hints[interval_var] = start_val
                #     model.AddHint(active_var, 0)
                #     model.AddHint(size_var, default_size)
                #     model.AddHint(start_var, start_val)
                #     model.AddHint(end_var, start_val + default_size)
                #     for _bool, *_ in vars["planet_terms"][star][i][0]:
                #         model.AddHint(_bool, 0)
                #     logger.debug(
                #         f"Added default hint for observation {n_obs} for star {star}"
                #     )

        for (star, planet), var in vars["all_planet_requested_observations"].items():
            star_df = prev_final_df[prev_final_df["star"] == star]
            # Use star_df['coeff0'], star_df['coeff1'], etc to set booleans for
            # number of detections
            meets_request = (
                star_df[f"coeff{planet}"] >= self.coeff_multiple * self.planet_threshold
            ).sum() >= self.requested_planet_observations
            model.AddHint(var, meets_request)
            # also set the above requested observations here
            above_request = (
                (
                    star_df[f"coeff{planet}"]
                    - (self.coeff_multiple * self.planet_threshold)
                )
                .sum()
                .astype(int)
            )
            above_var = vars["above_requested_observations"][(star, planet)]
            model.AddHint(above_var, above_request)
        # Leaving these here for now, may be useful later
        # for (inti, intj), var in vars["i_before_j_bools"].items():
        #     previ_start = interval_start_hints[inti]
        #     prevj_start = interval_start_hints[intj]
        #     i_before_j = previ_start < prevj_start
        #     model.AddHint(var, i_before_j)
        # for (int1_interval, int2_interval), (_bool, int1_start, int2_start) in vars[
        #     "int1_before_int2_bools"
        # ].items():
        #     prev1_start = interval_start_hints[int1_interval]
        #     prev2_start = interval_start_hints[int2_interval]
        #     int1_before_int2 = prev1_start < prev2_start
        #     model.AddHint(_bool, int1_before_int2)
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

    def add_percentages(self, flat_info, pdet):
        """
        Add the percentage of planets that have been observed at least once,
        twice, and three times to the flat_info dataframe as a function of time.
        This also re-orders the flat_info dataframe by time.
        """
        observable_planets = self.summary_stats["unique_planets_detected"]
        twice_observable_planets = self.summary_stats["two_detections"]
        thrice_observable_planets = self.summary_stats["three_plus_detections"]

        flat_info["time"] = Time(flat_info["obs_time"], format="jd").datetime
        flat_info = flat_info.sort_values("time").reset_index(drop=True)
        times = Time(flat_info["obs_time"], format="jd")
        flat_info["time_since_mission_start"] = (
            times.decimalyear - pdet.SS.TimeKeeping.missionStart.decimalyear
        )
        one_percents = []
        two_percents = []
        three_percents = []
        for i, obs in flat_info.iterrows():
            obs_counter = Counter(flat_info[: i + 1]["pind"])
            one_counter = sum([1 for k, v in obs_counter.items() if v >= 1])
            two_counter = sum([1 for k, v in obs_counter.items() if v >= 2])
            three_counter = sum([1 for k, v in obs_counter.items() if v >= 3])
            one_percents.append(one_counter / observable_planets)
            two_percents.append(
                two_counter / (twice_observable_planets + thrice_observable_planets)
            )
            three_percents.append(three_counter / thrice_observable_planets)
        flat_info["planets_detected_at_least_once"] = one_percents
        flat_info["planets_detected_at_least_twice"] = two_percents
        flat_info["planets_detected_at_least_thrice"] = three_percents
        return flat_info

    def schedule_plots(self, pdet):
        plt.style.use("dark_background")
        font = {"size": 13}
        plt.rc("font", **font)

        # Create histograms
        histfig, histaxes = plt.subplots(ncols=2, nrows=3, figsize=(8, 12))
        ax_names = [
            "unique_planets_detected",
            "one_detection",
            "two_detections",
            "three_plus_detections",
            "n_observations",
            "int_time",
        ]
        titles = [
            "Unique planets detected",
            "Planets with one detection",
            "Planets with two detections",
            "Planets with three+ detections",
            "Number of observations",
            "Summed integration time (d)",
        ]
        bin_iters = [1, 1, 1, 1, 2, 2]
        for ax, category, title, bin_iter in zip(
            histaxes.flatten(), ax_names, titles, bin_iters
        ):
            catvals = self.random_walk_res[category].values
            schval = self.summary_stats[category]
            bin_min = np.floor(catvals.min()) - 0.5
            bin_max = np.ceil(catvals.max()) + 0.5
            catbins = np.arange(bin_min, bin_max, bin_iter)
            ax.hist(catvals, bins=catbins, density=True)
            ax.axvline(schval, ls="--")
            ax.set_title(title)

        hist_path = Path(f"{self.result_path}/random_walk_comp.png")
        hist_path2 = Path(f"{str(self.finished_path)[:-2]}_comp.png")
        histfig.tight_layout()
        histfig.savefig(hist_path, dpi=300)
        histfig.savefig(hist_path2, dpi=300)

        SS = pdet.SS
        SU = SS.SimulatedUniverse
        start_time_jd = SS.TimeKeeping.missionStart.jd
        end_time_jd = start_time_jd + self.sim_length.to(u.d).value
        obs_times = Time(
            np.arange(start_time_jd, end_time_jd, self.block_length.to(u.d).value),
            format="jd",
            scale="tai",
        )
        figsc, axsc = plt.subplots(figsize=(11, 20))
        # Load the detection mode
        mode = list(
            filter(
                lambda mode: mode["detectionMode"],
                SS.TargetList.OpticalSystem.observingModes,
            )
        )[0]
        ko_sun = mode["syst"]["koAngles_Sun"].value.astype(int).tolist()
        koMaps = SS.koMaps[mode["syst"]["name"]]
        koTimes = SS.koTimes

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
        axsc.set_ylim([0, nplans + 1])
        axsc.set_yticks(np.arange(0, nplans + 1, 1))

        tick_labels = [""]
        blank_tick_labels = [""]
        self.total_int_time = self.flatdf.int_time.sum() * u.d
        for pind, label in enumerate(planet_names.values()):
            if pind in self.targetdf.columns:
                success = self.targetdf[pind]["success"]
                fail = self.targetdf[pind]["fail"]
                tick_labels.append(f"{success}/{success+fail}-{label}")
                self.total_success += success
                self.total_obs += success + fail
                self.per_planet_data.append([success, fail])
                if success > 0:
                    self.unique_planets_detected += 1
            else:
                tick_labels.append(f"{label}")
                self.per_planet_data.append([0, 0])
            blank_tick_labels.append(label)

        # tick_labels = list(planet_names.values())
        # tick_labels.insert(0, "")
        # tick_labels.append("")
        axsc.set_yticklabels(tick_labels)
        axsc.set_xlim([0, end_time_jd - start_time_jd])
        axsc.set_xlabel("Time since mission start (d)")

        axsc.set_title(
            f"ExoZodi quantile: {pdet.fEZ_quantile:.2f}, "
            f"Sun keepout: {ko_sun}, "
            f"{self.unique_planets_detected}/{self.total_planets} detected, "
            f"{self.total_success}/{self.total_obs}, "
            f"{self.total_int_time}"  # , "
            # r"$\sigma_{RV}$"
            # f"={self.best_precision} m/s"  # , "
        )
        for yval in range(nplans + 2):
            axsc.axhline(y=yval + 0.5, ls="-", lw=0.25, zorder=1)

        # pdet_int_time = (
        #     np.median(self.block_multiples) * self.block_length.to(u.d).value
        # )
        cmap = plt.get_cmap("viridis")
        # figsc2, axsc2 = plt.subplots(figsize=(11, 20))
        # axsc2.set_ylim([0, nplans + 1])
        # axsc2.set_yticks(np.arange(0, nplans + 1, 1))
        # axsc2.set_yticklabels(tick_labels)
        # axsc2.set_xlim([0, end_time_jd - start_time_jd])
        # axsc2.set_xlabel("Time since mission start (d)")

        # axsc2.set_title(
        #     f"ExoZodi quantile: {pdet.fEZ_quantile:.2f}, "
        #     f"{self.unique_planets_detected}/{self.total_planets} detected, "
        #     f"{self.total_success}/{self.total_obs}, "
        #     f"{self.total_int_time} d"  # , "
        #     # r"$\sigma_{RV}$"
        #     # f"={self.best_precision} m/s"  # , "
        # )
        # cbar_2 = figsc.colorbar(
        #     mpl.cm.ScalarMappable(cmap=cmap),
        #     ax=axsc,
        #     alpha=0.5,
        #     location="right",
        #     ticks=np.linspace(0, 1, 5),
        # )
        # cbar_2.set_label(
        #     r"$P_{det}(t_{int}=$", fontsize=15
        # )  # + f"{pdet_int_time:.0f}d)", fontsize=15)
        det_colors = {0: "red", 1: "lime", -1: "yellow", -2: "yellow"}
        det_colors_schedule = copy.deepcopy(det_colors)
        det_colors_schedule[1] = "white"
        n_inds = 50
        plot_times = obs_times[::10]
        dt = 10 * self.block_length.to(u.d).value
        SS.reset_sim(genNewPlanets=False)
        used_sInds = []
        ####
        thresh_arr = np.zeros((nplans, len(self.block_multiples)))
        ####
        for system_name in tqdm(pdet.pops.keys(), desc="Generating plots"):
            system_pdets = pdet.pdets[system_name]

            pops = pdet.pops[system_name]
            for pval, pop in enumerate(pops):
                self.planets_fitted += 1
                planet_pdet = system_pdets.pdet[pval]
                pdet_vals = planet_pdet.interp(
                    time=obs_times.datetime,
                    int_time=np.array(self.block_multiples)
                    * self.block_length.to(u.d).value,
                ).values
                pop = pdet.pops[system_name][pval]
                sInd = np.where(SS.TargetList.Name == system_name)[0][0]
                pInds = np.where(SU.plan2star == sInd)[0]
                pInd = pInds[np.argmin(np.abs(np.median(pop.a) - SU.a[pInds]))]
                extent = 0, end_time_jd - start_time_jd, pInd + 0.5, pInd + 1.5

                # This is the threshold for the probability of detection
                alphas = np.ones(pdet_vals.shape) * 0.3
                # This sets the alpha based on the detection threshold
                # alphas[pdet_vals >= self.planet_threshold] = 0.75

                # This sets it based on keepout
                target_ko = np.interp(obs_times.jd, koTimes.jd, koMaps[sInd]).astype(
                    bool
                )
                alphas[:, target_ko] = 0.75

                # Have to find the right pInd to plot on since it's not
                # in the same order
                axsc.imshow(
                    pdet_vals,
                    aspect="auto",
                    interpolation="none",
                    extent=extent,
                    cmap=cmap,
                    norm=mpl.colors.Normalize(0, 1),
                    alpha=alphas,
                    zorder=0,
                )
                ####
                for _blockn, _ in enumerate(self.block_multiples):
                    thresh_arr[pInd, _blockn] = sum(
                        pdet_vals[_blockn, :] > self.planet_threshold
                    )
                ####
                # axsc2.imshow(
                #     pdet_vals,
                #     aspect="auto",
                #     interpolation="none",
                #     extent=extent,
                #     cmap=cmap,
                #     norm=mpl.colors.Normalize(0, 1),
                #     alpha=alphas,
                #     zorder=0,
                # )
                if system_name not in self.schedule.star.values:
                    continue
                fig, (ax_WA, ax_dMag) = plt.subplots(figsize=(11, 5), ncols=2)
                # sInd = np.where(SS.TargetList.Name == system_name)[0][0]
                # pInds = np.where(SU.plan2star == sInd)[0]
                # pInd = pInds[np.argmin(np.abs(np.median(pop.a) - SU.a[pInds]))]
                if pInd not in self.targetdf.columns:
                    continue
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
                ax_dMag.plot(
                    plot_times.jd - plot_times[0].jd, edMags, color="white", lw=0.75
                )
                ax_WA.plot(
                    plot_times.jd - plot_times[0].jd, eWAs, color="white", lw=0.75
                )

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
                        zorder=2,
                        # color="white",
                        edgecolor=det_colors_schedule[_det_status],
                        hatch=r"\\",
                        alpha=1,
                        fill=False,
                    )
                    axsc.add_patch(obs_sq)
                    # axsc.axvline(x=zeroed_time, alpha=0.25)
                    # axsc.axvline(x=zeroed_time + _tint, alpha=0.25, ls="--")

                # fEZstr = f"{self.targetdf[pInd]['fEZ'][nobs].value:.0e}"
                intstr = intstr[:-2]
                SNRstr = SNRstr[:-2]

                ax_dMag.set_ylabel(r"$\Delta$mag")
                ax_WA.set_ylabel('Planet-star angular separation (")')
                ax_WA.set_xlabel("Time since mission start (d)")
                ax_dMag.set_xlabel("Time since mission start (d)")
                fig.suptitle(
                    f"{system_name}, "
                    f"RV precision: {self.best_precision} m/s, "
                    # f"int time: [{intstr}] d, "
                    f"SNRs: [{SNRstr}]"
                    # f"fEZ: {fEZstr}"
                )
                fig.tight_layout()
                fig.savefig(fig_path, dpi=300)
        ####
        pd.DataFrame(thresh_arr, columns=self.block_multiples)
        # breakpoint()
        # threshdf
        ####
        figsc_path1 = Path(f"{self.result_path}/full_schedule.png")
        # figsc_path2 = Path(f"{self.finished_path}ng")
        figsc.tight_layout()
        figsc.savefig(figsc_path1, dpi=300)
        # figsc.savefig(figsc_path2, dpi=300)

        # figsc2_path1 = Path("dissertation_plots/pdettradespace.png")
        # figsc2.tight_layout()
        # figsc2.savefig(figsc2_path1, dpi=300)
