import pickle
from pathlib import Path

import astropy.units as u
import numpy as np

from rvsearch import search
from RVtools.logger import logger


class OrbitFit:
    """
    Base class to do orbit fitting.
    """

    def __init__(self, params, preobs, universe, workers):
        self.method = params["fitting_method"]
        self.max_planets = params["max_planets"]
        self.systems_to_fit = params["systems_to_fit"]
        self.dynamic_max = params["dynamic_max"]
        self.workers = workers

        self.paths = {}
        self.planets_fitted = {}
        if self.method == "rvsearch":
            self.use_rvsearch(preobs, universe)

    def use_rvsearch(self, preobs, universe):
        """
        This method takes in the precursor observation object and the universe
        object to run orbit fitting with the RVsearch tool.
        """
        for i, system_id in enumerate(self.systems_to_fit):
            rv_df = preobs.syst_observations[system_id]
            system = universe.systems[system_id]
            star_name = system.star.name
            # breakpoint()
            system_path = f"{universe.cache_path.parents[0]}/rvsearch/{star_name}"
            Path(system_path).mkdir(exist_ok=True, parents=True)
            if self.dynamic_max:
                # Determine the maximum number of planets that can be detected
                k_vals = system.getpattr("K")
                # Assuming that the semi-amplitude has to be 10 times larger than the
                # best instrument's precision
                k_cutoff = 10 * min([inst.precision for inst in preobs.instruments])
                feasible_planets = np.where(k_vals > k_cutoff)[0]
                feasible_max = len(feasible_planets)
                max_planets = min([feasible_max, self.max_planets])
            else:
                max_planets = self.max_planets
            logger.info(
                (
                    f"Searching {star_name} for up to {max_planets} planets."
                    f"Star {i+1} of {len(self.systems_to_fit)}."
                )
            )
            searcher = search.Search(
                rv_df,
                starname=star_name,
                workers=self.workers,
                mcmc=True,
                verbose=True,
                max_planets=max_planets,
                mstar=(system.star.mass.to(u.M_sun).value, 0),
            )
            searcher.run_search(outdir=system_path)
            search_path = Path(system_path, "search.pkl")
            with open(search_path, "rb") as f:
                sch = pickle.load(f)
            planets_fitted = sch.post.params.num_planets
            logger.info(f"Found {planets_fitted} planets around {star_name}.")
            self.paths[system_id] = system_path
            self.planets_fitted[system_id] = planets_fitted
