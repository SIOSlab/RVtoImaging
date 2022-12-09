from random import choice, expovariate, sample

import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

from RVtools.logger import logger


class PreObs:
    """
    Base class for precursor observations.
    Main attributes are
    syst_observations (dict):
        Dictionary that holds all observations of a single stellar system in DataFrames
    observations (Pandas DataFrame):
        DataFrame with all observations
    """

    def __init__(self, params, universe):
        """
        params (dict):
            Parameters used by the PreObs module
        universe (Universe):
            Universe object that observations will be taken on
        """

        if "base_params" in params:
            base_params = params["base_params"]
        else:
            base_params = {}

        insts = params["instruments"]
        self.instruments = []
        for inst in insts:
            # Start with common params then add user specified instrument
            # parameters
            inst_params = base_params.copy()
            inst_params.update(inst)

            # Instantiate instrumentes
            instrument = Instrument(inst_params)

            # Create the times available for observation
            logger.info(f"Generating rv times for {inst['name']}")
            instrument.generate_available_rv_times()

            # Keep track of instruments
            self.instruments.append(instrument)

        logger.info("Assigning observations")
        # We assign observation times to the systems of interest
        self.systems_to_observe = params["systems_to_observe"]
        for inst in self.instruments:
            inst.assign_instrument_observations(self.systems_to_observe)

        # Calculate all the times the systems will need to be propagated
        logger.info("Getting propagation times")
        self.get_propagation_times()

        # Propagate systems to all necessary times
        logger.info("Propagating systems")
        self.propagate_systems(universe)

        # Now we have to actually make the observations
        for inst in self.instruments:
            inst.make_observations(universe)

        # Combine the observations into one dataframe
        for i, inst in enumerate(self.instruments):
            inst_observations = inst.observations
            if i == 0:
                observations = inst_observations
            else:
                observations = pd.concat(
                    [observations, inst_observations], ignore_index=True
                )
        self.observations = observations

        # orbit fitting software friendly system observation dataframes
        self.syst_observations = {}
        for system_id in self.systems_to_observe:
            self.syst_observations[system_id] = observations.loc[
                observations.system_id == system_id
            ].reset_index(drop=True)

    def get_propagation_times(self):
        """
        This gets the times at which we need to propagate the systems to
        simulate observations with all instruments. It stores them in
        self.prop_times, a dictionary where the system id is the key to the
        necessary propagation times.
        """
        self.prop_times = {}
        for system_id in self.systems_to_observe:
            system_observations = []
            for inst in self.instruments:
                # Get the times where an observation of the current system_id
                # will be made with this instrument
                inst_obs_inds = np.where(inst.observation_schedule == system_id)[0]
                obs_times = inst.rv_times[inst_obs_inds]
                system_observations.append(obs_times.jd)
            self.prop_times[system_id] = Time(
                np.sort(np.concatenate(system_observations)), format="jd"
            )

    def propagate_systems(self, universe):
        """
        This calls each system's propagation function with the relevant
        propagation times.
        """
        for system_id in self.systems_to_observe:
            system = universe.systems[system_id]
            times = self.prop_times[system_id]
            system.propagate(times)


class Instrument:
    """
    This class will hold an instrument's parameters, e.g. observation times and
    observation values
        inst_params values:
            instrument_precisions (list of astropy Quantities):
                Each instrument's precision in velocity units e.g. m/s
            type (str):
                'fixed'
                    The times are provided
                    requires:
                        set_times (astropy Time array)
                'Poisson'
                    Times should be calculated with a Poisson process
                    requires:
                        start_time (astropy Time)
                            - First available time for observation
                        end_time (astropy Time)
                            - Last available time for observation
                        rates (list of astropy rate Quantities - e.g. time^(-1))
                            - Rate of observations for each instrument
                'equal'
                    Give a start and end time and a number of observations then
                    distributes them equally
                    requires:
                        start_time (astropy Time quantity)
                            - First available time for observation
                        end_time (astropy Time quantity)
                            - Last available time for observation
                        num (integer)
                            - Number of observations
            observation_scheme (str):
                'time_cluster' - Cluster observations based on cluster_length
                    requires:
                        cluster_length (astropy Quantity e.g. 5 days)
                            - Time spent observing one target before changing
                        cluster_choice (str)
                            'cycle' - Schedules system observations in a cycle
                            'random' - Chooses next system to observe randomly
                TODO:
                    'all' - Observe each system at each available time
                    'cycle' - Cycle through each system and observe them in order
                    'random' - Observe the systems randomly
    """

    def __init__(self, inst_params):

        self.name = inst_params["name"]
        self.precision = inst_params["precision"]

        # Observation time generation parameters
        self.timing_format = inst_params["timing_format"]
        if self.timing_format == "fixed":
            self.rv_times = inst_params["set_times"]
        elif self.timing_format == "Poisson":
            self.start_time = inst_params["start_time"]
            self.end_time = inst_params["end_time"]
            self.rate = inst_params["rate"]
        elif self.timing_format == "equal":
            self.start_time = inst_params["start_time"]
            self.end_time = inst_params["end_time"]
            self.num = inst_params["num"]

        # Observation assignment parameters
        self.observation_scheme = inst_params["observation_scheme"]
        if self.observation_scheme == "time_cluster":
            self.cluster_length = inst_params["cluster_length"]
            self.cluster_choice = inst_params["cluster_choice"]
            self.targets_per_observation = 1
        elif self.observation_scheme == "survey":
            self.targets_per_observation = inst_params["targets_per_observation"]

    def generate_available_rv_times(self):
        """
        Based on the timing format set up the rv times available for
        observation
        """
        if self.timing_format == "fixed":
            # If fixed then we already set it in init
            pass

        elif self.timing_format == "Poisson":
            current_time = Time(self.start_time.jd, format="jd")
            times_list = []
            time_is_valid = True
            while time_is_valid:
                # Generate spacing between observations with expovariate
                # function to match a Poisson process
                time_until_next_observation = expovariate(self.rate)
                current_time += time_until_next_observation

                # Check if we've exceeded the maximum time allowed
                if current_time >= self.end_time:
                    time_is_valid = False
                else:
                    times_list.append(current_time.jd)

            # Save to final format
            self.rv_times = Time(times_list, format="jd")

        elif self.timing_format == "equal":
            # Simple linear spacing
            self.rv_times = np.linspace(self.start_time, self.end_time, self.num)

        # Standardize formatting into decimalyear
        self.rv_times.format = "decimalyear"

    def assign_instrument_observations(self, systems_to_observe):
        """
        This will assign each of the instrument's observation time to a specific
        star.
        """
        # This array is holds the indices of the system(s) that will be
        # observed at each available rv time
        self.observation_schedule = np.zeros(
            (len(self.rv_times), self.targets_per_observation), dtype=int
        )
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
            # Observe every star at each observation time
            for i, observation in enumerate(self.rv_times):
                # Choose the targets that will be observed at every observation time
                systems = sample(systems_to_observe, self.targets_per_observation)
                for j, system_id in enumerate(systems):
                    self.observation_schedule[i, j] = system_id

        # # Save schedule instrument
        # self.observation_schedule = observation_schedule

    def make_observations(self, universe):
        """
        Simulate the process of making observations for an instrument.
        """
        observed_systems = np.unique(self.observation_schedule)
        logger.info(f"Simulating RV observations for {self.name}")
        for system_id in tqdm(observed_systems, desc="Observing stars"):
            # Need to keep track of which instrument observations are on the
            # current system, and which system rv_vals those observations
            # correspond to

            # Start by getting the times when this instrument is observing
            # the current system
            inst_obs_inds = np.where(self.observation_schedule == system_id)[0]
            rv_obs_times = self.rv_times[inst_obs_inds]

            # Now get the system's true rv values at those observation times
            system = universe.systems[system_id]
            df = system.rv_df[system.rv_df.t.isin(rv_obs_times)]
            true_rv_SI = df.rv

            # Instrument's precision in matching units
            inst_precision_SI = self.precision.decompose().value

            # Get the velocity offsets assuming Gaussian noise
            rv_offset = np.random.normal(scale=inst_precision_SI, size=len(true_rv_SI))
            observed_rv = true_rv_SI + rv_offset

            # Formatting to create DataFrame with all relevant values
            t_jd = rv_obs_times.jd
            t_decimalyear = rv_obs_times.decimalyear
            precision_array = np.repeat(inst_precision_SI, len(rv_obs_times))
            system_array = np.repeat(system_id, len(rv_obs_times))
            inst_array = np.repeat(self.name, len(rv_obs_times))
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
                    precision_array,
                    inst_array,
                    true_rv_SI,
                    t_decimalyear,
                    system_array,
                ),
                axis=-1,
            )
            obs_df = pd.DataFrame(stacked_arrays, columns=columns)

            # Simplify datatypes
            dtypes = {
                "time": np.float,
                "mnvel": np.float,
                "errvel": np.float,
                "tel": str,
                "truevel": np.float,
                "t_year": np.float,
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
