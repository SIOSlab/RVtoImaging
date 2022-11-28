import astropy.units as u
import numpy as np
from astropy.time import Time


class PreObs:
    """
    Base class for precursor observations.
    """

    def __init__(self, params):
        """
        params values:
            type (str):
                'fixed' - The times are provided
                    requires:
                        set_times (astropy Time Quantity array)
                'Poisson' - Times should be calculated with a Poisson process
                    requires:
                        start_time (astropy Time quantity)
                            - First available time for observation
                        end_time (astropy Time quantity)
                            - Last available time for observation
                        rate (float)
                            - Number of observations per month
                'equal' - Give a start and end time and a number of
                            observations then distributes them equally
                    requires:
                        start_time (astropy Time quantity)
                            - First available time for observation
                        end_time (astropy Time quantity)
                            - Last available time for observation
                        num (integer)
                            - Number of observations
                ''
        """
        self.type = params["type"]
        if self.type == "fixed":
            self.times = params["set_times"]
        elif self.type == "Poisson":
            self.start_time = params["start_time"]
            self.end_time = params["end_time"]
            self.rate = params["rate"]
        elif self.type == "equal":
            self.start_time = params["start_time"]
            self.end_time = params["end_time"]
            self.num = params["num"]
            self.times = np.linspace(self.start_time, self.end_time, self.num)

    def poisson_times(self):
        total_time = self.end_time.decimalyear - self.start_time.decimalyear
        months_available = int(total_time * 12)
        raw_schedule = np.random.poisson(lam=self.rate, size=months_available)

        # Go through the raw schedule and assign observations
        times = np.zeros(sum(raw_schedule))
        obs_n = 0  # This keeps track of what the current observation is
        for month_num, obs_in_month in enumerate(raw_schedule):
            if obs_in_month == 0:
                continue
            # Get the month to assign values within
            time_period = self.start_time + [month_num, month_num + 1] * u.yr / 12
            times_in_month = np.random.uniform(
                time_period[0].value, time_period[1].value, size=obs_in_month
            )
            for obs_time in times_in_month:
                # Add each observation in this month to the observation array
                times[obs_n] = Time(obs_time, format="decimalyear").jd
                obs_n += 1
        # times[0] = self.start_time.jd
        # times[-1] = self.end_time.jd
        self.times = times
