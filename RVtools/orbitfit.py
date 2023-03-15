import pickle
import time
from datetime import datetime
from pathlib import Path

import astropy.units as u
from astropy.time import Time

import RVtools.utils as utils
from rvsearch import search
from RVtools.cosmos import FitSystem
from RVtools.logger import logger


class OrbitFit:
    """
    Base class to do orbit fitting.
    """

    def __init__(self, params, universe, surveys, workers):
        self.workers = workers
        # self.method = params["fitting_method"]
        # self.max_planets = params["max_planets"]
        # self.vary_planets = params["vary_planets"]
        # self.universe_dir = Path(params["universe_dir"])
        for param, param_value in params.items():
            # save input parameters to this object
            setattr(self, param, param_value)

        # self.systems_to_fit = params["systems_to_fit"]
        self.paths = []
        self.planets_fitted = {}

        # Sort surveys by fit order
        # for survey_ind in fit_order:
        #     survey = surveys[survey_ind]
        self.use_rvsearch(universe, surveys)

    def use_rvsearch(self, universe, surveys):
        """
        This method takes in the precursor observation object and the universe
        object to run orbit fitting with the RVsearch tool.
        """
        # fit_order = np.argsort([survey.fit_order for survey in surveys])
        self.fits_completed = 0
        self.fits_loaded = 0
        start_time = time.time()
        for survey in surveys:
            self.systems_to_fit = survey.systems_to_observe
            for i, system_id in enumerate(self.systems_to_fit):
                rv_df = survey.syst_observations[system_id]
                system = universe.systems[system_id]
                star_name = system.star.name
                # This could be uncommented, but with such low search times for
                # new planets I don't think it's necessary

                # Determine the maximum number of planets that can be detected
                # k_vals = system.getpattr("K")

                # Assuming that the semi-amplitude has to be 10 times larger than the
                # best instrument's precision and max period is 75 years
                # k_cutoff = 3 * min([inst.precision for inst in survey.instruments])
                # feasible_max = sum(
                #     (k_vals > k_cutoff) & (system.getpattr("T") < 75 * u.yr)
                # )
                # max_planets = min([feasible_max, self.max_planets])
                max_planets = self.max_planets

                # Handle caching of fits, structure is that each system
                system_path = f"{self.universe_dir}/{star_name}"
                # Directory to save fit based on max number of planets ("depth")
                survey_path = Path(system_path, survey.name)
                fit_dir = Path(survey_path, f"{max_planets}_depth")
                self.paths.append(fit_dir)
                if max_planets == 0:
                    logger.warning(f"No detections feasible around {star_name}.")
                    continue

                # Check the folder for previous fits
                has_fit, prev_max, fitting_done, _ = utils.check_orbitfit_dir(
                    survey_path
                )
                # best_candidate_fit = utils.prev_best_fit(system_path, survey.name)

                # Path(system_path).mkdir(exist_ok=True)

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
                            previous_dir = Path(survey_path, f"{prev_max}_depth")
                            logger.info(
                                (
                                    f"Loading previous fit information on {star_name} "
                                    f"from {previous_dir}. New search max is "
                                    f"{max_planets}."
                                )
                            )
                            # Load previous search
                            with open(Path(previous_dir, "search.pkl"), "rb") as f:
                                searcher = pickle.load(f)

                            # Set new maximum planets
                            searcher.max_planets = max_planets
                    # elif (best_candidate_fit["folder"] is not None) and (
                    #     best_candidate_fit["prev_max"] > 1
                    # ):
                    #     # Second best scenario is if another fit has been done with
                    #     # worse data that we can start with and then improve by using
                    #     # better data
                    #     logger.info(
                    #         (
                    #             f"Loading previous fit information on {star_name} "
                    #             f"from {best_candidate_fit['search_path']}."
                    #         )
                    #     )
                    #     searcher = self.change_post(
                    #         survey,
                    #         rv_df,
                    #         max_planets,
                    #         best_candidate_fit["search_path"],
                    #     )
                    else:
                        searcher = search.Search(
                            rv_df,
                            starname=star_name,
                            workers=self.workers,
                            mcmc=True,
                            verbose=True,
                            max_planets=max_planets,
                            mstar=(system.star.mass.to(u.M_sun).value, 0),
                            n_vary=self.vary_planets,
                        )

                    if hasattr(self, "total_searches"):
                        # When running seeds we use a different calculation
                        # that accounts for how long the previous universes
                        # took to calculate
                        current_time = time.time()
                        runs_left = (
                            self.total_searches
                            - self.loaded_searches
                            - self.completed_searches
                            - i
                        )
                        # Initial start time set in builder.run_seeds
                        elapsed_time = current_time - self.initial_start_time
                        if (self.fits_completed + self.completed_searches) > 0:
                            # Rate at which one fit has been completed
                            rate = elapsed_time / (
                                self.fits_completed + self.completed_searches
                            )
                            finish_time = datetime.fromtimestamp(
                                current_time + rate * runs_left
                            )
                            finish_str = finish_time.strftime("%c")
                            rate_str = f"{rate/60:.2f} minutes per search"
                        else:
                            finish_str = "TBD"
                            rate_str = "TBD"
                        universe_str = (
                            f"in universe {self.universe_number} of"
                            f"{self.total_universes}. "
                        )
                    elif self.fits_completed > 0:
                        current_time = time.time()
                        runs_left = len(self.systems_to_fit) - i
                        elapsed_time = current_time - start_time
                        rate = elapsed_time / self.fits_completed
                        finish_time = datetime.fromtimestamp(
                            current_time + rate * runs_left
                        )
                        finish_str = finish_time.strftime("%c")
                        rate_str = f"{rate/60:.2f} minutes per search"
                        universe_str = "in universe 1 of 1. "
                    else:
                        finish_str = "TBD"
                        rate_str = "TBD"
                        universe_str = "in universe 1 of 1. "
                    logger.info(
                        (
                            f"Searching {star_name} for up to {max_planets} planets. "
                            f"Star {i+1} of {len(self.systems_to_fit)} "
                            f"{universe_str}"
                            f"{rate_str}. "
                            f"Estimated finish: {finish_str}."
                        )
                    )

                    # Run search
                    searcher.run_search(outdir=str(fit_dir), running=False)
                    self.fits_completed += 1

                    n_obs = rv_df.shape[0]
                    obs_baseline = (
                        (
                            Time(max(rv_df.time), format="jd")
                            - Time(min(rv_df.time), format="jd")
                        )
                        .to(u.yr)
                        .value
                    )
                    if searcher.mcmc_failure:
                        fit_spec = {
                            "max_planets": int(max_planets),
                            "planets_fitted": 0,
                            "mcmc_converged": False,
                            "observations": int(n_obs),
                            "observational_baseline": obs_baseline,
                            "mcmc_success": False,
                        }
                        logger.warning(f"Failure to run MCMC on {star_name}.")
                    else:
                        # Save specifications of the orbit fit
                        planets_fitted = searcher.post.params.num_planets
                        fit_spec = {
                            "max_planets": int(max_planets),
                            "planets_fitted": int(planets_fitted),
                            "mcmc_converged": bool(searcher.mcmc_converged),
                            "observations": int(n_obs),
                            "observational_baseline": obs_baseline,
                            "mcmc_success": True,
                        }
                        logger.info(
                            f"Found {planets_fitted} planets around {star_name}."
                        )
                        if searcher.num_planets > 0:
                            # Create a system from the fitted planets
                            fitted_system = FitSystem(searcher, system)
                            with open(Path(fit_dir, "fitsystem.p"), "wb") as f:
                                pickle.dump(fitted_system, f)

                    # Save specs
                    utils.update(fit_dir, fit_spec)
                else:
                    self.fits_loaded += 1
