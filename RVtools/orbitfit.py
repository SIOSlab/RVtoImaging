from pathlib import Path

import astropy.units as u

from rvsearch import search


class OrbitFit:
    """
    Base class to do orbit fitting.
    """

    def __init__(self, params, preobs, universe):
        self.method = params["fitting_method"]
        self.max_planets = params["max_planets"]
        if self.method == "rvsearch":
            self.use_rvsearch(preobs, universe)

    def use_rvsearch(self, preobs, universe):
        for system_id in preobs.systems_to_observe:
            rv_df = preobs.syst_observations[system_id]
            system = universe.systems[system_id]
            star_name = system.star.name
            system_path = f".cache/{universe.hash}/{star_name}"
            Path(system_path).mkdir(exist_ok=True, parents=True)
            searcher = search.Search(
                rv_df,
                starname=star_name,
                # min_per=min_period.value,
                # oversampling=10,
                # max_per=min(max_period.value, 10000),
                workers=14,
                mcmc=True,
                verbose=True,
                max_planets=self.max_planets,
                mstar=(system.star.mass.to(u.M_sun).value, 0),
            )
            searcher.run_search(outdir=system_path, fixed_threshold=500)
