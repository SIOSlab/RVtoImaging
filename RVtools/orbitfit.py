import pickle
import time
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time

import radvel
import RVtools.utils as utils
from rvsearch import search
from RVtools.logger import logger


class OrbitFit:
    """
    Base class to do orbit fitting.
    """

    def __init__(self, params, library, universe, surveys, workers):
        self.method = params["fitting_method"]
        self.max_planets = params["max_planets"]
        self.workers = workers
        self.universe_dir = Path(params["universe_dir"])

        # self.systems_to_fit = params["systems_to_fit"]
        self.paths = {}
        self.planets_fitted = {}

        # Sort surveys by fit order
        # for survey_ind in fit_order:
        #     survey = surveys[survey_ind]
        self.use_rvsearch(library, universe, surveys)

    def use_rvsearch(self, library, universe, surveys):
        """
        This method takes in the precursor observation object and the universe
        object to run orbit fitting with the RVsearch tool.
        """
        # fit_order = np.argsort([survey.fit_order for survey in surveys])
        for survey in surveys:
            start_time = time.time()
            fits_completed = 0
            systems_to_fit = survey.systems_to_observe
            for i, system_id in enumerate(systems_to_fit):
                rv_df = survey.syst_observations[system_id]
                system = universe.systems[system_id]
                star_name = system.star.name
                # Determine the maximum number of planets that can be detected
                k_vals = system.getpattr("K")

                # Assuming that the semi-amplitude has to be 10 times larger than the
                # best instrument's precision and max period is 75 years
                k_cutoff = 3 * min([inst.precision for inst in survey.instruments])
                feasible_max = sum(
                    (k_vals > k_cutoff) & (system.getpattr("T") < 75 * u.yr)
                )
                max_planets = min([feasible_max, self.max_planets])
                if max_planets == 0:
                    logger.warning(f"No detections feasible around {star_name}.")
                    continue

                # Handle caching of fits, structure is that each system
                system_path = f"{self.universe_dir}/{star_name}"
                # Directory to save fit based on max number of planets ("depth")
                survey_path = Path(system_path, survey.name)
                has_fit, prev_max, fitting_done, _ = utils.check_orbitfit_dir(
                    survey_path
                )
                best_candidate_fit = utils.prev_best_fit(system_path, survey.name)

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
                    elif (best_candidate_fit["folder"] is not None) and (
                        best_candidate_fit["max_planets"] > 1
                    ):
                        # Second best scenario is if another fit has been done with
                        # worse data that we can start with and then improve by using
                        # better data
                        logger.info(
                            (
                                f"Loading previous fit information on {star_name} "
                                f"from {best_candidate_fit['search_path']}."
                            )
                        )
                        searcher = self.change_post(
                            survey,
                            rv_df,
                            max_planets,
                            best_candidate_fit["search_path"],
                        )
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

                    fit_dir = Path(survey_path, f"{max_planets}_depth")
                    if fits_completed > 0:
                        current_time = time.time()
                        runs_left = len(systems_to_fit) - i
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
                            f" Star {i+1} of {len(systems_to_fit)}. "
                            f"Estimated finish for orbit fitting: {finish_str}"
                        )
                    )

                    # Run search
                    searcher.run_search(outdir=str(fit_dir), running=False)
                    fits_completed += 1

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

                    # Save specs
                    library.update(fit_dir, fit_spec)

    def change_post(self, survey, rv_df, max_planets, search_path):
        """
        To change the post we have to update the data and reset gamma/jitter values
        """
        with open(search_path, "rb") as f:
            searcher = pickle.load(f)
        searcher.data = rv_df
        searcher.workers = self.workers
        searcher.max_planets = max_planets
        initial_params = searcher.post.params
        vector = searcher.post.vector.vector
        vector_names = searcher.post.vector.names
        vector_indices = searcher.post.vector.indices
        param_keys = [key for key in initial_params.keys()]
        deleted_keys = 0
        for key in param_keys:
            if "gamma" in key or "jit" in key:
                # Remove it from the names, indices, and row
                value = vector_indices[key]
                del vector_names[value - deleted_keys]
                vector = np.delete(vector, (value - deleted_keys), axis=0)
                vector_indices.pop(key)
                deleted_keys += 1
                initial_params.pop(key)
        next_ind = vector.shape[0]

        gamma_arr = np.array([[0, 0, 0, 1]])
        jit_arr = np.array([[2, 1, 0, 0]])
        extra_params = []
        for inst in survey.instruments:
            # Add gamma
            gamma_key = f"gamma_{inst.name}"
            vector = np.concatenate([vector, gamma_arr], axis=0)
            vector_names.append(gamma_key)
            vector_indices[gamma_key] = next_ind
            initial_params[gamma_key] = radvel.Parameter(
                value=0.0, linear=True, vary=False
            )
            extra_params.append(gamma_key)
            next_ind += 1

            # Add jitter
            jit_key = f"jit_{inst.name}"
            vector = np.concatenate([vector, jit_arr], axis=0)
            vector_names.append(jit_key)
            vector_indices[jit_key] = next_ind
            initial_params[jit_key] = radvel.Parameter(value=0.0)
            extra_params.append(jit_key)
            next_ind += 1
        searcher.post.params = initial_params
        searcher.post.vector.vector = vector
        searcher.post.vector.names = vector_names
        searcher.post.vector.indices = vector_indices
        searcher.post.likelihood.extra_params = extra_params

        # Change the vary parameters
        # no_vary_keys = ["per", "secosw", "sesinw"]
        # vary_keys = ["k", "tc"]
        # for planet in range(searcher.num_planets):
        # pnum = searcher.num_planets + 1
        # for key in no_vary_keys:
        #     vind = searcher.post.vector.indices[f"{key}{pnum}"]
        #     searcher.post.vector.vector[vind, 1] = 0
        # for key in vary_keys:
        #     vind = searcher.post.vector.indices[f"{key}{pnum}"]
        #     searcher.post.vector.vector[vind, 1] = 1
        # breakpoint()
        searcher.post.vector.vector_to_dict()
        searcher.post.list_vary_params()
        searcher.post.get_vary_params()

        # Subtract until there's one planet less than the max
        while (
            searcher.post.params.num_planets >= max_planets
            or searcher.post.params.num_planets == 1
        ):
            print("subtracted planet")
            searcher.sub_planet()
        # searcher.post = radvel.fitting.maxlike_fitting(searcher.post, verbose=True)
        return searcher
