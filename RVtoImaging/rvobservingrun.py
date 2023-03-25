import hashlib
import json
from math import modf
from pathlib import Path
from random import choice, random, sample

import astropy.units as u
import dill
import numpy as np
import pandas as pd
from astroplan import Observer, observability_table
from astropy.time import Time
from tqdm import tqdm

from RVtoImaging.logger import logger


class RVObservingRun:
    """
    This class will hold the parameters necessary for an observing run, e.g.
    observation times, rv precision terms, and constraints
    """

    def __init__(self, run_params, universe):
        # Observing run name
        self.name = run_params["name"]

        # Dictionary with the observation scheduling information for this run
        self.obs_scheme = run_params["observation_scheme"]

        # The terms used to calculate sigma_rv fot this observing run
        self.sigma_terms = run_params["sigma_terms"]

        # Survey start/end times
        self.start_time = run_params["start_time"]
        self.end_time = run_params["end_time"]

        # Must be a site name from astropy.coordinates.EarthLocation.get_site_names(),
        # or None if using the simple observing scheme
        self.location = run_params["location"]

        if "target_df" in run_params.keys():
            self.target_df = run_params["target_df"]

        # Create spec file for dumping later
        self.spec = {
            "start_time": self.start_time.decimalyear,
            "end_time": self.end_time.decimalyear,
            "location": self.location,
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
        self.run_path = Path(
            run_params["universe_dir"], "obs_runs", f"{self.runhash}.p"
        )
        if self.run_path.exists():
            with open(self.run_path, "rb") as f:
                prev_calc = dill.load(f)
            self.target_df = prev_calc.target_df
            self.observations = prev_calc.observations
        else:
            # Create astroplan observer object at location
            self.observer = Observer.at_site(self.location)

            # Create observation schedule
            self.create_observation_schedule()

            # Simulate the observations
            self.make_observations(universe)

            # Cache, ensuring the folder exists
            self.run_path.parent.mkdir(exist_ok=True)
            with open(self.run_path, "wb") as f:
                dill.dump(self, f)

    def create_observation_schedule(self):
        """
        Creates an observation schedule according to the observation scheme set
        """
        # Calculate which nights the stars are observable given the time frame
        # Maybe set this up with the times value instead of time_range?
        logger.info("Creating observability table for the RV target stars.")
        self.observability_table = observability_table(
            self.obs_scheme["astroplan_constraints"],
            self.observer,
            self.target_df.target.tolist(),
            time_range=[self.start_time, self.end_time],
        )

        # Create the years available so that we can assign observations
        # in them according to the observations_per_star_per_year parameter
        # note: modf splits a float 2.13 into (.13, 2.0)
        intermediate_years = np.arange(
            modf(self.start_time.decimalyear)[1] + 1, modf(self.end_time.decimalyear)[1]
        )

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
        year_blocks = pd.DataFrame(
            {"start": dates[:-1].decimalyear, "end": dates[1:].decimalyear}
        )

        # calculate the number of observations per year, most years
        # will have nobs observations, but this accounts for the first
        # and last years
        nobs = self.obs_scheme["observations_per_star_per_year"]
        year_blocks["observations"] = round(
            (year_blocks["end"] - year_blocks["start"]) * nobs
        )

        # The times observability is calculated at
        times = Time(self.observability_table.meta["times"])
        times_jd = times.jd
        times_year = times.decimalyear
        all_observations = []

        # Create the observation schedule
        match (self.obs_scheme["type"]):
            case "simple":
                # Create observation schedule for the targets
                logger.info(f"Creating observation times for {self.name} targets")
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
                    target_observations = np.zeros(int(year_blocks.observations.sum()))

                    n_obs = 0
                    for _, block in year_blocks.iterrows():
                        # Get the observation times inside this block (observing year)
                        in_block = (block.end > times_year) & (
                            times_year >= block.start
                        )

                        # The array of observations to be made in this block
                        block_observations = Time(
                            np.zeros(int(block.observations)), format="jd"
                        )

                        # This keeps track of which observation times are too
                        # close to an already scheduled observation
                        time_removed = np.zeros(len(times), dtype=bool)

                        for obs_ind in range(int(block.observations)):
                            # Get array of observations available
                            available_observation_inds = np.where(
                                target_observability["always observable"]
                                & in_block
                                & ~time_removed
                            )[0]

                            # Select from those inds, checking that we aren't
                            # doing multiple observations on the same night
                            observation_time = Time(
                                times_year[choice(available_observation_inds)],
                                format="decimalyear",
                            )
                            block_observations[obs_ind] = observation_time

                            # Now to ensure that we aren't scheduling
                            # observations on the same night remove them by
                            # making all observations within 12 hours of the
                            # new one have True values in the time_removed
                            # array
                            close_inds = np.where(
                                np.abs(observation_time.jd - times_jd) < 0.5
                            )[0]
                            time_removed[close_inds] = True
                        for observation in sorted(block_observations):
                            target_observations[n_obs] = observation.jd
                            n_obs += 1
                    all_observations.append(Time(target_observations, format="jd"))
                self.target_df["observations"] = all_observations
            case "constraint":
                # Set up which nights are available for observing
                self.schedule_observing_nights()

                # Determine target observability for all possible nights
                self.calc_observability()

    def schedule_observing_nights(self):
        # In more complicated observing schemes we schedule the nights
        # first and then assign observations based on them
        breakpoint()

    def generate_available_rv_times(self):
        """
        Based on the timing format set up the rv times available for
        observation
        """
        # if self.timing_format == "fixed":
        #     # If fixed then we already set it in init
        #     pass

        # elif self.timing_format == "Poisson":
        #     current_time = Time(self.start_time.jd, format="jd")
        #     times_list = []
        #     time_is_valid = True
        #     while time_is_valid:
        #         # Generate spacing between observations with expovariate
        #         # function to match a Poisson process
        #         time_until_next_observation = expovariate(self.rate)
        #         current_time += time_until_next_observation

        #         # Check if we've exceeded the maximum time allowed
        #         if current_time >= self.end_time:
        #             time_is_valid = False
        #         else:
        #             times_list.append(current_time.jd)

        #     # Save to final format
        #     self.rv_times = Time(times_list, format="jd")

        # elif self.timing_format == "equal":
        #     # Simple linear spacing
        #     self.rv_times = np.linspace(self.start_time, self.end_time, self.num)

        nights_available = int(self.end_time.jd - self.start_time.jd)

        # Generate random numbers to represent bad weather
        # If the random number is less than the bad weather probability
        # then it's considered a bad weather night and no observations
        # will be taken
        bad_weather_nights = (
            np.array([random() for _ in range(int(nights_available))])
            < self.bad_weather_prob
        )

        # Set observations to start at 10 pm and finish at 4 am
        offset_to_obs_start = 10 - 24 * (modf(self.start_time.jd)[0])
        first_obs_time = self.start_time + (offset_to_obs_start) / 24 * u.d
        # last_obs_of_night_time = first_obs_time + 6 * u.hr
        obs_spacing = np.linspace(0, 6 / 24, self.observations_per_night)
        self.obs_nights = []
        times_arr = np.array([])
        for night, bad_weather in zip(range(nights_available), bad_weather_nights):
            if not bad_weather:
                # Set up times for the night
                self.obs_nights.append(night)
                times_arr = np.append(
                    times_arr, (first_obs_time + night + obs_spacing).jd
                )

        self.rv_times = Time(times_arr, format="jd")
        # Standardize formatting into decimalyear
        self.rv_times.format = "decimalyear"

    def assign_instrument_observations(self, systems_to_observe):
        """
        This will assign each of the instrument's observation time to a specific
        star.
        """
        # This array is holds the indices of the system(s) that will be
        # observed at each available rv time
        self.observation_schedule = np.zeros(len(self.rv_times), dtype=int)
        if self.observation_scheme == "time_cluster":
            # Necessary parameters
            cluster_start_time = self.rv_times[0]
            current_system_ind = 0

            # Used when cycling
            n_systems = len(systems_to_observe)

            # Loop over all the available observation times
            for i, observation in enumerate(self.rv_times):
                # Time since the cluster started
                elapsed_time = observation - cluster_start_time

                # Check whether it's time to move to the next cluster of
                # observations (e.g. it's been 1.1 days and a cluster is only 1
                # day)
                if elapsed_time > self.cluster_length:
                    # Choosing next system to observe
                    if self.cluster_choice == "cycle":
                        current_system_ind = (current_system_ind + 1) % n_systems
                    elif self.cluster_choice == "random":
                        current_system_ind = choice(systems_to_observe)

                    # Make the current time the start time of the next cluster
                    cluster_start_time = observation

                self.observation_schedule[i, 0] = current_system_ind
        elif self.observation_scheme == "survey":
            # Observe a random set of stars each night
            for i, _ in enumerate(self.obs_nights):
                # Choose the targets that will be observed at every observation time
                systems = sample(systems_to_observe, self.observations_per_night)
                self.observation_schedule[
                    i
                    * self.observations_per_night : self.observations_per_night
                    * (1 + i)
                ] = systems
        # breakpoint()
        # for system_id in systems:
        #     self.observation_schedule[i] = system_id

        # # Save schedule instrument
        # self.observation_schedule = observation_schedule

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
