import pickle
import time
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time

from rvsearch import search
from RVtools.logger import logger


class OrbitFit:
    """
    Base class to do orbit fitting.
    """

    def __init__(self, params, library, universe, preobs, workers):
        self.method = params["fitting_method"]
        self.max_planets = params["max_planets"]
        self.workers = workers
        self.cache_dir = Path(params["cache_dir"])

        self.systems_to_fit = np.array(preobs.systems_to_observe)[
            params["systems_to_fit"]
        ]
        self.paths = {}
        self.planets_fitted = {}
        if self.method == "rvsearch":
            self.use_rvsearch(library, universe, preobs)

    def use_rvsearch(self, library, universe, preobs):
        """
        This method takes in the precursor observation object and the universe
        object to run orbit fitting with the RVsearch tool.
        """
        start_time = time.time()
        fits_completed = 0
        for i, system_id in enumerate(self.systems_to_fit):
            rv_df = preobs.syst_observations[system_id]
            system = universe.systems[system_id]
            star_name = system.star.name
            # if self.dynamic_max:
            # Determine the maximum number of planets that can be detected
            k_vals = system.getpattr("K")

            # Assuming that the semi-amplitude has to be 10 times larger than the
            # best instrument's precision and max period is 35 years
            k_cutoff = 10 * min([inst.precision for inst in preobs.instruments])
            feasible_max = sum((k_vals > k_cutoff) & (system.getpattr("T") < 35 * u.yr))
            max_planets = min([feasible_max, self.max_planets])
            if max_planets == 0:
                logger.warning(f"No detections feasible around {star_name}.")
                continue

            # Handle caching of fits, structure is that each system
            system_path = f"{self.cache_dir}/{star_name}"
            # Path(system_path).mkdir(exist_ok=True)

            # Directory to save fit based on max number of planets ("depth")
            has_fit, prev_max, fitting_done = library.check_orbitfit_dir(system_path)
            if not fitting_done:
                # If a full fit has been done then no futher progress can be made
                if has_fit:
                    if max_planets <= prev_max:
                        logger.info(
                            (
                                f"Previous fit attempt is the same or better "
                                f"for {star_name}. No orbit fitting necessary."
                            )
                        )
                        continue
                    else:
                        previous_dir = Path(system_path, f"{prev_max}_depth")
                        logger.info(
                            (
                                f"Loading previous fit information on {star_name} "
                                f"from {previous_dir}. New search max is {max_planets}."
                            )
                        )
                        # Load previous search
                        with open(Path(previous_dir, "search.pkl"), "rb") as f:
                            searcher = pickle.load(f)

                        # Set new maximum planets
                        searcher.max_planets = max_planets
                else:
                    searcher = search.Search(
                        rv_df,
                        starname=star_name,
                        workers=self.workers,
                        mcmc=True,
                        verbose=True,
                        max_planets=max_planets,
                        mstar=(system.star.mass.to(u.M_sun).value, 0),
                    )

                fit_dir = Path(system_path, f"{max_planets}_depth")
                if fits_completed > 0:
                    current_time = time.time()
                    runs_left = len(self.systems_to_fit) - i
                    elapsed_time = current_time - start_time
                    rate = elapsed_time / fits_completed
                    finish_time = datetime.fromtimestamp(
                        current_time + rate * runs_left
                    )
                    finish_str = finish_time.strftime("%c")
                else:
                    finish_str = "TBD"
                logger.info(
                    (
                        f"Searching {star_name} for up to {max_planets} planets."
                        f" Star {i+1} of {len(self.systems_to_fit)}. "
                        f"Estimated finish for orbit fitting: {finish_str}"
                    )
                )

                # Run search
                searcher.run_search(outdir=str(fit_dir))

                # Save specifications of the orbit fit
                planets_fitted = searcher.post.params.num_planets
                n_obs = rv_df.shape[0]
                obs_baseline = (
                    (
                        Time(max(rv_df.time), format="jd")
                        - Time(min(rv_df.time), format="jd")
                    )
                    .to(u.yr)
                    .value
                )
                fit_spec = {
                    "max_planets": int(max_planets),
                    "planets_fitted": int(planets_fitted),
                    "mcmc_converged": bool(searcher.mcmc_converged),
                    "observations": int(n_obs),
                    "observational_baseline": obs_baseline,
                }
                fits_completed += 1

                # Save specs
                library.update(fit_dir, fit_spec)
                logger.info(f"Found {planets_fitted} planets around {star_name}.")
