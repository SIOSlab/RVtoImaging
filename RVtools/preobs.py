from random import choice, expovariate

import numpy as np
from astropy.time import Time


class PreObs:
    """
    Base class for precursor observations.
    """

    def __init__(self, params, universe):
        """
        system_ids (list of int):
            The integer indices of the systems to observe
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
            instrument.generate_available_rv_times()

            # Keep track of instruments
            self.instruments.append(instrument)

        # Now we need to actually simulate the observations
        systems_to_observe = params["systems_to_observe"]
        for inst in self.instruments:
            inst.assign_instrument_observations(systems_to_observe)
        breakpoint()


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
        if self.observation_scheme == "time_cluster":
            # Necessary parameters
            cluster_start_time = self.rv_times[0]
            current_system_ind = 0

            # Used when cycling
            n_systems = len(systems_to_observe)

            # This list is the index of the system that will be observed at
            # each available rv time
            observation_schedule = []

            # Loop over all the available observation times
            for observation in self.rv_times:
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

                observation_schedule.append(current_system_ind)

        # Save schedule instrument
        self.observation_schedule = observation_schedule
