from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time


class PDet:
    """
    Base class to do probability of detection calculations
    """

    def __init__(self, params, orbitfit, universe):
        self.method = params["construction_method"]
        self.systems_of_interest = params["systems_of_interest"]

        self.pops = {}

        # Loop through all the systems we want to calculate probability of
        # detection for
        for system_id in self.systems_of_interest:
            if system_id in orbitfit.paths.keys():
                system_path = orbitfit.paths[system_id]
                # Check for the chains
                chains_path = Path(system_path, "chains.csv.tar.bz2")
                if chains_path.exists():
                    # chains = pd.read_csv(chains_path)
                    system = universe.systems[system_id]
                    self.pops[system_id] = PlanetPopulation(orbitfit, system)
                else:
                    print(f"No chains were created for system {system_id}")
            else:
                print(f"No orbit fitting was attempted for system {system_id}")

            system = universe.systems[system_id]
            self.pops[system_id] = PlanetPopulation(orbitfit, system)


class PlanetPopulation:
    """
    Class that holds constructed orbits that could have created the RV data
    """

    def __init__(self, chains, system):
        """
        Args:
            chains (pandas.DataFrame):
                The MCMC chains generated from the orbit fitting process
            system (System):
                The system data
        """

        self.samples_for_cov = (
            chains.pipe(self.start_pipeline)
            .pipe(self.sort_by_lnprob)
            .pipe(self.get_samples_for_covariance)
            .pipe(self.drop_columns)
        )
        self.cov_df = self.samples_for_cov.cov()
        self.chains_means = (
            chains.pipe(self.start_pipeline).pipe(self.drop_columns).mean()
        )
        chain_samples_np = np.random.multivariate_normal(
            self.chains_means, self.cov_df, size=self.n_fits
        )
        chain_samples = pd.DataFrame(chain_samples_np, columns=self.cov_df.keys())

        # Use those samples and assign the values
        self.T = chain_samples.per1.to_numpy() * u.d
        self.secosw = chain_samples.secosw1.to_numpy()
        self.sesinw = chain_samples.sesinw1.to_numpy()
        self.K = chain_samples.k1.to_numpy()
        self.T_c = Time(chain_samples.tc1.to_numpy(), format="jd")

    def get_samples_for_covariance(self, df):
        return df.head(self.cov_samples)

    def sort_by_lnprob(self, df):
        return df.sort_values("lnprobability", ascending=False)

    def drop_columns(self, df):
        return df.drop(columns=self.droppable_cols)
