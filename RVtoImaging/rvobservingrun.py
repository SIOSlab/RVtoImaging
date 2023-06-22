import hashlib
import json
from math import modf
from pathlib import Path
from random import choice, random, sample

import astropy.units as u
import dill
import numpy as np
import pandas as pd
import pytz
from astroplan import Observer, observability_table
from astropy.time import Time
from ortools.sat.python import cp_model
from tqdm import tqdm

from RVtoImaging.logger import logger


class RVObservingRun:
    """
    This class will hold the parameters necessary for an observing run, e.g.
    observation times, rv precision terms, and constraints
    """

    def __init__(self, run_params, universe, workers):
        # Observing run name
        self.name = run_params["name"].replace(" ", "_").replace("/", "")

        # Dictionary with the observation scheduling information for this run
        self.obs_scheme = run_params["observation_scheme"]

        # The terms used to calculate sigma_rv fot this observing run
        self.sigma_terms = run_params["sigma_terms"]

        # Must be a site name from astropy.coordinates.EarthLocation.get_site_names(),
        # or None if using the simple observing scheme
        self.location = run_params["location"]
        # Easiest way to use this is to run
        # TimeQuantity.to_datetime(pytz.timezone('utc')).astimezone(self.timezone)
        self.timezone = pytz.timezone(run_params["timezone"])

        # Survey start/end times
        self.start_time = Time(
            self.timezone.localize(run_params["start_time"].to_datetime())
        )
        self.end_time = Time(
            self.timezone.localize(run_params["end_time"].to_datetime())
        )
        # self.start_time = run_params["start_time"]
        # self.end_time = run_params["end_time"]

        if "target_df" in run_params.keys():
            self.target_df = run_params["target_df"]

        # Create spec file for dumping later
        self.spec = {
            "start_time": self.start_time.decimalyear,
            "end_time": self.end_time.decimalyear,
            "location": self.location,
            "timezone": run_params["timezone"],
        }

        # Add the sigma terms
        sigma_terms_decomposed = {}
        for sigma_term, sigma_val in self.sigma_terms.items():
            sigma_terms_decomposed[sigma_term] = sigma_val.decompose().value
            self.spec[sigma_term] = sigma_val.decompose().value
        # self.spec["sigma_terms"] = sigma_terms_decomposed

        # observation_scheme_dict = {}
        for obs_scheme_term, obs_scheme_val in self.obs_scheme.items():
            if type(obs_scheme_val) == u.Quantity:
                val = obs_scheme_val.decompose().value
            elif obs_scheme_term == "astroplan_constraints":
                val = [str(type(constraint)) for constraint in obs_scheme_val]
            else:
                val = obs_scheme_val
            # observation_scheme_dict[obs_scheme_term] = val
            self.spec[obs_scheme_term] = val
        # self.spec['observation_scheme'] = observation_scheme_dict

        # Check whether this scheme has already been created
        self.runhash = hashlib.sha1(
            json.dumps(self.spec, sort_keys=True).encode("UTF-8")
        ).hexdigest()[:8]
        self.run_name = self.name + "_" + self.runhash
        self.run_path = Path(
            run_params["universe_dir"], "obs_runs", f"{self.run_name}.p"
        )
        if self.run_path.exists():
            logger.info(f"Loading the {self.name} observing run from {self.run_path}")
            with open(self.run_path, "rb") as f:
                prev_calc = dill.load(f)
            self.target_df = prev_calc.target_df
            self.observations = prev_calc.observations
        else:
            # Create astroplan observer object at location
            self.observer = Observer.at_site(self.location, timezone=self.timezone)

            # Create observation schedule
            self.create_observation_schedule(workers)

            # Simulate the observations
            self.make_observations(universe)

            # Cache, ensuring the folder exists
            self.run_path.parent.mkdir(exist_ok=True)
            with open(self.run_path, "wb") as f:
                dill.dump(self, f)

    def create_observation_schedule(self, workers):
        """
        Creates an observation schedule according to the observation scheme set
        Adds all observation times to the self.target_df DataFrame
        """
        # Calculate which nights the stars are observable given the time frame
        # Maybe set this up with the times value instead of time_range?
        logger.info(
            "Creating observability table for the RV target stars from "
            f"{self.start_time.decimalyear:.2f} to {self.end_time.decimalyear:.2f} "
            f"for {self.name} observing run."
        )
        self.observability_table = observability_table(
            self.obs_scheme["astroplan_constraints"],
            self.observer,
            self.target_df.target.tolist(),
            time_range=[self.start_time, self.end_time],
            time_grid_resolution=self.obs_scheme["exposure_time"],
        )

        # Create the years available so that we can assign observations
        # in them according to the observations_per_star_per_year parameter
        # note: modf splits a float 2.13 into (.13, 2.0)
        intermediate_years = np.arange(
            modf(self.start_time.decimalyear)[1] + 1, modf(self.end_time.decimalyear)[1]
        )

        # Localize to the timezone, man do I hate timezones
        intermediate_years = [
            Time(
                self.timezone.localize(
                    Time(year, format="decimalyear", scale="utc").to_datetime()
                )
            ).decimalyear
            for year in intermediate_years
        ]

        # Array that goes [2000.5, 2001, 2002, ..., 2014, 2014.2] for a
        # observing run with start_time=2000.5 and end_time = 2014.2
        dates = Time(
            np.hstack(
                (
                    self.start_time.decimalyear,
                    intermediate_years,
                    self.end_time.decimalyear,
                )
            ),
            format="decimalyear",
        )
        self.year_blocks = pd.DataFrame(
            {"start": dates[:-1].decimalyear, "end": dates[1:].decimalyear}
        )

        # The times observability is calculated at
        times = Time(self.observability_table.meta["times"])
        times_jd = times.jd
        times_year = times.decimalyear
        all_observations = []
        logger.info(f"Creating observation times for {self.name} targets")
        # Create the observation schedule
        match (self.obs_scheme["type"]):
            case "constraint":
                # Set up which nights are available for observing
                obs_nights = self.schedule_observing_nights()

                # Get the indices of all available observation slots created
                # in the observability table
                all_obs_night_inds = []
                tracking_the_night = []
                for night in obs_nights:
                    night_inds = np.where(np.abs(times.jd - night.jd) < 0.5)[0]
                    for ind in night_inds:
                        all_obs_night_inds.append(ind)
                        tracking_the_night.append(night)
                obs_df = pd.DataFrame(
                    {
                        "observation_inds": all_obs_night_inds,
                        "nights": tracking_the_night,
                    }
                )

                # Remove inds that have no targets detectable
                rows_to_delete = []
                for row_ind, (obs_ind, night) in obs_df.iterrows():
                    # obs_time = times[obs_ind]
                    if ~np.any(
                        self.observability_table["always observable"][:, obs_ind]
                    ):
                        rows_to_delete.append(row_ind)
                        # useful_inds.append(obs_ind)
                        # useful_inds_nights.append(night)
                reduced_obs_df = obs_df.drop(index=rows_to_delete).reset_index(
                    drop=True
                )

                # This is a horrific 'oneliner' that creates a dictionary where every
                # key is a night and the value for the key is an array of all
                # the time indices that correspond to a time when an observation can be
                # made on that night
                night_slots_dict = {
                    night: reduced_obs_df.loc[
                        reduced_obs_df.nights == night
                    ].observation_inds.values
                    for night in np.unique(reduced_obs_df.nights)
                }

                # Goal is to now make the best schedule we can
                model = cp_model.CpModel()

                # Making iterating easier
                targets = self.target_df.HIP.values
                valid_nights = list(night_slots_dict.keys())

                # Create the decision variables
                obs_assignments = {}
                for target in targets:
                    for night in valid_nights:
                        for slot in night_slots_dict[night]:
                            obs_assignments[(target, night, slot)] = model.NewBoolVar(
                                f"{target}_{night.jd:.2f}_{slot}"
                            )

                # Make sure each observation is only assigned to at most one target
                for night in valid_nights:
                    for slot in night_slots_dict[night]:
                        model.AddAtMostOne(
                            obs_assignments[(target, night, slot)] for target in targets
                        )

                # Make sure each target is observed at most once per night
                for target in targets:
                    for night in valid_nights:
                        model.AddAtMostOne(
                            obs_assignments[(target, night, slot)]
                            for slot in night_slots_dict[night]
                        )

                # maximize the number of targets with desired number of observations
                n_requested_observations = self.obs_scheme["requested_observations"]
                requested_observation_bools = []
                for target in targets:
                    _bool = model.NewBoolVar(
                        (
                            f"{target} has at least "
                            f"{n_requested_observations} observations"
                        )
                    )
                    target_bools = []
                    for night in valid_nights:
                        for slot in night_slots_dict[night]:
                            target_bools.append(obs_assignments[(target, night, slot)])
                    model.Add(
                        sum(target_bools) >= n_requested_observations
                    ).OnlyEnforceIf(_bool)
                    model.Add(
                        sum(target_bools) < n_requested_observations
                    ).OnlyEnforceIf(_bool.Not())
                    requested_observation_bools.append(_bool)

                # Objective function, maximize the number of observations
                # made subject to the previous constraints and the number
                # of target stars that have at least the number of observations
                # requested
                model.Maximize(
                    sum(
                        obs_assignments[(target, night, slot)]
                        for target in targets
                        for night in valid_nights
                        for slot in night_slots_dict[night]
                    )
                    + 100 * sum(requested_observation_bools)
                )

                logger.info(f"Creating optimal observation schedule for {self.name}")
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = workers
                solver.parameters.log_search_progress = self.obs_scheme[
                    "log_search_progress"
                ]
                solver.parameters.max_time_in_seconds = self.obs_scheme[
                    "max_time_in_seconds"
                ]
                solver.Solve(model)
                # if status == cp_model.OPTIMAL:
                n_target_observations = []
                for target in targets:
                    target_observations = []
                    for night in valid_nights:
                        for slot in night_slots_dict[night]:
                            if (
                                solver.Value(obs_assignments[(target, night, slot)])
                                == 1
                            ):
                                target_observations.append(times_jd[slot])
                    n_target_observations.append(len(target_observations))
                    all_observations.append(Time(target_observations, format="jd"))
                self.target_df["n_observations"] = n_target_observations
                self.target_df["observations"] = all_observations

                targets_observed = sum((self.target_df.n_observations.values > 1))
                logger.info(
                    f"Optimal schedule found, observing {targets_observed} "
                    "stars at least once."
                )
                # breakpoint()
                # else:
                #     raise RuntimeError(
                #         f"No optimal schedule possible for {self.name} observing run."
                #     )
            case "random":
                # Calculate the number of observations per year, most years
                # will have nobs observations, but this accounts for the first
                # and last years
                nobs = self.obs_scheme["observations_per_star_per_year"]
                self.year_blocks["n_observations"] = round(
                    (self.year_blocks["end"] - self.year_blocks["start"]) * nobs
                )
                for _, target in tqdm(
                    self.target_df.iterrows(),
                    total=len(self.target_df),
                    desc="Target",
                    position=0,
                ):
                    # get the index of the target in the observability_table,
                    # should be the same index but this feels more safe
                    obs_table_ind = np.where(
                        self.observability_table["target name"].data == target.HIP
                    )[0][0]

                    # Find the times availble based on the observability_table
                    target_observability = self.observability_table[obs_table_ind]

                    # List to keep track of all the observations
                    # target_observations = np.zeros(
                    #     int(year_blocks.n_observations.sum())
                    # )
                    target_observations = []

                    n_obs = 0
                    for _, block in self.year_blocks.iterrows():
                        # Get the observation times inside this block (observing year)
                        in_block = (block.end > times_year) & (
                            times_year >= block.start
                        )

                        # The array of observations to be made in this block
                        block_observations = []

                        # This keeps track of which observation times are too
                        # close to an already scheduled observation
                        time_removed = np.zeros(len(times), dtype=bool)

                        for obs_ind in range(int(block.n_observations)):
                            # Get array of observations available
                            available_observation_inds = np.where(
                                target_observability["always observable"]
                                & in_block
                                & ~time_removed
                            )[0]
                            if len(available_observation_inds) > 0:
                                # Select from those inds, checking that we aren't
                                # doing multiple observations on the same night
                                observation_time = Time(
                                    times_year[choice(available_observation_inds)],
                                    format="decimalyear",
                                )
                                block_observations.append(observation_time.jd)

                                # Now to ensure that we aren't scheduling
                                # observations on the same night remove them by
                                # making all observations within 12 hours of the
                                # new one have True values in the time_removed
                                # array
                                close_inds = np.where(
                                    np.abs(observation_time.jd - times_jd) < 0.5
                                )[0]
                                time_removed[close_inds] = True
                            else:
                                break

                        # Randomly calculate bad weather and remove
                        # observations based on it
                        bad_weather_nights = (
                            np.array([random() for _ in range(len(block_observations))])
                            < self.obs_scheme["bad_weather_prob"]
                        )
                        block_observations = np.array(block_observations)[
                            ~bad_weather_nights
                        ]

                        # Add the observations to the target_observation list
                        for observation in sorted(block_observations):
                            target_observations.append(observation)
                            n_obs += 1

                    all_observations.append(Time(target_observations, format="jd"))
                self.target_df["observations"] = all_observations

    def schedule_observing_nights(self):
        """
        Create a schedule of which nights are to be observed each year
        and return it as

        Note: a night is specified at midnight (assuming the initial start time
        is at midnight) for self.timezone but stored as utc. A night extends for
        12 hours in either direction. So the night of 2038-01-01 00:00:00 includes
        all times between 2037-12-31 12:00:00 - 2038-01-01 12:00:00.
        """
        # In more complicated observing schemes we schedule the nights
        # first and then assign observations based on them
        obs_nights = []
        if self.obs_scheme["obs_night_schedule"] == "random":
            # For the random case we randomly sample from all nights in the
            # year according to the specified "n_obs_nights" parameter
            for _, block in self.year_blocks.iterrows():
                start_quantity = Time(block.start, format="decimalyear")
                end_quantity = Time(block.end, format="decimalyear")
                block_dates = np.arange(start_quantity.jd, end_quantity.jd, 1)

                block_obs_nights = sample(
                    sorted(block_dates), self.obs_scheme["n_obs_nights"]
                )
                for obs in sorted(block_obs_nights):
                    obs_nights.append(obs)
            obs_nights = Time(obs_nights, format="jd")
        else:
            raise ValueError("This night scheduling scheme doesn't exist")
        return obs_nights

    def make_observations(self, universe):
        """
        Simulate the process of making observations for an instrument.
        """
        # observed_systems = np.unique(self.observation_schedule)
        logger.info(f"Simulating RV observations for {self.name}")
        for _, target in tqdm(
            self.target_df.iterrows(),
            total=self.target_df.shape[0],
            desc="Observing stars",
        ):
            # Need to keep track of which instrument observations are on the
            # current system, and which system rv_vals those observations
            # correspond to
            system_id = int(target.universe_id)

            # Start by getting the times when this instrument is observing
            # the current system
            rv_obs_times = target.observations
            if len(rv_obs_times) == 0:
                continue

            # Now get the system's true rv values at those observation times
            system = universe.systems[system_id]
            system.propagate(rv_obs_times)
            df = system.rv_df[system.rv_df.t.isin(rv_obs_times)]
            true_rv_SI = df.rv

            # RVInstrument's sigma_rv in matching units
            run_sigma_rv_SI = self.sigma_rv(system, rv_obs_times).decompose().value

            # Get the velocity offsets assuming Gaussian noise
            rv_offset = np.random.normal(scale=run_sigma_rv_SI, size=len(true_rv_SI))
            observed_rv = true_rv_SI + rv_offset

            # Formatting to create DataFrame with all relevant values
            t_jd = rv_obs_times.jd
            t_decimalyear = rv_obs_times.decimalyear
            sigma_rv_array = np.repeat(run_sigma_rv_SI, len(rv_obs_times))
            system_array = np.repeat(system_id, len(rv_obs_times))
            run_array = np.repeat(self.name, len(rv_obs_times))
            columns = [
                "time",
                "mnvel",
                "errvel",
                "tel",
                "truevel",
                "t_year",
                "system_id",
            ]
            stacked_arrays = np.stack(
                (
                    t_jd,
                    observed_rv,
                    sigma_rv_array,
                    run_array,
                    true_rv_SI,
                    t_decimalyear,
                    system_array,
                ),
                axis=-1,
            )
            obs_df = pd.DataFrame(stacked_arrays, columns=columns)

            # Simplify datatypes
            # dtypes = {
            #     "time": np.float,
            #     "mnvel": np.float,
            #     "errvel": np.float,
            #     "tel": str,
            #     "truevel": np.float,
            #     "t_year": np.float,
            #     "system_id": int,
            # }
            dtypes = {
                "time": float,
                "mnvel": float,
                "errvel": float,
                "tel": str,
                "truevel": float,
                "t_year": float,
                "system_id": int,
            }
            obs_df = obs_df.astype(dtypes)

            # Concat or create observations DataFrame
            if hasattr(self, "observations"):
                self.observations = pd.concat(
                    [self.observations, obs_df], ignore_index=True
                )
            else:
                self.observations = obs_df

        # Sort the observations by time
        self.observations = self.observations.sort_values(by="time").reset_index(
            drop=True
        )

    def sigma_rv(self, system, times):
        # TODO make this more intelligent
        sigma_values = np.fromiter(self.sigma_terms.values(), dtype=u.Quantity)
        sigma_rv = np.sqrt(np.mean(np.square(sigma_values)))
        return sigma_rv
