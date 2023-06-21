import json
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astroplan import FixedTarget
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.simbad import Simbad

from RVtoImaging.rvobservingrun import RVObservingRun


class RVDataset:
    """
    Base class for the RV dataset
    Main attributes are
    syst_observations (dict):
        Dictionary that holds all observations of a single stellar system in DataFrames
    observations (Pandas DataFrame):
        DataFrame with all observations
    """

    def __init__(self, params, universe, workers):
        """
        params (dict):
            Parameters used by the RVDataset module
        universe (Universe):
            Universe object that observations will be taken on
        """

        if fname := params.get("available_targets_file"):
            using_preset_targets = True
            with open(fname, "rb") as f:
                available_targets = pd.read_csv(f)
            # Fill in missing target star information
            target_df = self.fill_stellar_information(available_targets, universe)
        else:
            using_preset_targets = False
            # FILL IN WITH DEFAULT SETTINGS
            pass

        obs_run_names = []
        self.obs_runs = []
        for obs_run_params in params["rv_observing_runs"]:
            if using_preset_targets:
                obs_run_params["target_df"] = target_df
            obs_run_params["universe_dir"] = params["universe_dir"]
            _obs_run = RVObservingRun(obs_run_params, universe, workers)
            obs_run_names.append(_obs_run.run_name)
            self.obs_runs.append(_obs_run)

        self.name = "runs_"
        for run_name in sorted(obs_run_names):
            self.name += run_name

        # Combine the observations into one dataframe
        for i, obs_run in enumerate(self.obs_runs):
            obs_run_observations = obs_run.observations
            if i == 0:
                observations = obs_run_observations
            else:
                observations = pd.concat(
                    [observations, obs_run_observations], ignore_index=True
                )
        self.observations = observations

        # orbit fitting software friendly system observation dataframes
        self.syst_observations = {}
        for system_id in np.unique(self.observations.system_id):
            syst_obs = observations.loc[
                observations.system_id == system_id
            ].reset_index(drop=True)
            self.syst_observations[system_id] = syst_obs

            # Save the observations for each system in their folder
            syst_folder = Path(
                params["universe_dir"],
                f"{universe.names[system_id].replace(' ', '_')}",
                self.name,
            )
            syst_folder.mkdir(exist_ok=True, parents=True)

            # Save the specification for the observations in a json for easier
            # parsing
            obs_spec = {}
            obs_spec["rv_dataset"] = params["dataset_name"]
            obs_spec["number_of_observations_all_obs_runs"] = syst_obs.shape[0]
            # obs_spec["obs_baseline_all_instruments"] = (
            #     syst_obs.iloc[np.argmax(syst_obs.t_year)].t_year
            #     - syst_obs.iloc[np.argmin(syst_obs.t_year)].t_year
            # )
            obs_run_specs = {}
            earliest_start = None
            latest_end = None
            for obs_run in self.obs_runs:
                obs_run_spec = obs_run.spec
                obs_run_spec["number_of_observations"] = syst_obs.loc[
                    syst_obs.tel == obs_run.name
                ].shape[0]
                obs_run_specs[obs_run.name] = obs_run_spec

                if earliest_start is None:
                    earliest_start = obs_run_spec["start_time"]
                    latest_end = obs_run_spec["end_time"]
                if obs_run_spec["start_time"] < earliest_start:
                    earliest_start = obs_run_spec["start_time"]
                if obs_run_spec["end_time"] > latest_end:
                    latest_end = obs_run_spec["end_time"]
            obs_spec["obs_baseline"] = latest_end - earliest_start
            obs_spec["obs_runs"] = obs_run_specs

            # Save spec
            with open(Path(syst_folder, "obs_spec.json"), "w") as f:
                json.dump(obs_spec, f)

            # Save raw data
            syst_obs.to_csv(Path(syst_folder, "rv.csv"))

    def fill_stellar_information(self, tldf, universe):
        """
        Method to fill in missing information, right now it is not generalized
        and programmed for the NETS list specifically
        """
        Simbad.add_votable_fields("ids")
        if "HIP" not in tldf.columns:
            # Get the HIP ids and resave the target list
            hip_ids = []
            in_universe = []
            universe_ids = []
            coords = []
            targets = []

            for name in tldf.name:
                query = Simbad.query_object(name)
                all_ids = query["IDS"][0].split("|")
                coord = SkyCoord(
                    ra=query["RA"].data.data,
                    dec=query["DEC"].data.data,
                    unit=(u.hourangle, u.deg, u.arcsec),
                )
                coords.append(coord)
                if "HIP" in query["IDS"][0]:
                    simbad_hip_value = [id for id in all_ids if "HIP" in id][0]
                    # Doing string surgery to match convention of a single
                    # space between HIP and the number
                    sp = simbad_hip_value.split()
                    hip_id = f"{sp[0]} {sp[1]}"
                    hip_ids.append(hip_id)
                    in_universe_val = hip_id in universe.names
                    in_universe.append(in_universe_val)
                    if in_universe_val:
                        universe_ids.append(
                            np.where(np.array(universe.names) == hip_id)[0][0]
                        )
                    else:
                        universe_ids.append(None)
                    targets.append(FixedTarget(coord=coord, name=hip_id))
                else:
                    hip_ids.append(None)
                    in_universe.append(False)
                    universe_ids.append(None)
                    targets.append(None)
                # dist = query["Distance_distance"].data.data * u.pc
                # print(simbad_list['RA'].data.data)

            tldf["coordinate"] = coords
            tldf["target"] = targets
            tldf["HIP"] = hip_ids
            tldf["in_universe"] = in_universe
            tldf["universe_id"] = universe_ids
            target_df = tldf.loc[tldf.in_universe].reset_index(drop=True)
            # systems_to_observe = target_df.universe_id.astype(int).tolist()
            return target_df

    def choose_systems(self, universe):
        Simbad.add_votable_fields("ids")
        if self.target_list is not None:
            tldf = pd.read_csv(self.target_list)
            if "HIP" not in tldf.columns:
                # Get the HIP ids and resave the target list
                hip_ids = []
                in_universe = []
                universe_ids = []

                for name in tldf.name:
                    query = Simbad.query_object(name)
                    all_ids = query["IDS"][0].split("|")
                    if "HIP" in query["IDS"][0]:
                        simbad_hip_value = [id for id in all_ids if "HIP" in id][0]
                        # Doing string surgery to match convention of a single
                        # space between HIP and the number
                        sp = simbad_hip_value.split()
                        hip_id = f"{sp[0]} {sp[1]}"
                        hip_ids.append(hip_id)
                        in_universe_val = hip_id in universe.names
                        in_universe.append(in_universe_val)
                        if in_universe_val:
                            universe_ids.append(
                                np.where(np.array(universe.names) == hip_id)[0][0]
                            )
                        else:
                            universe_ids.append(None)
                    else:
                        hip_ids.append(None)
                        in_universe.append(False)
                        universe_ids.append(None)
                tldf["HIP"] = hip_ids
                tldf["in_universe"] = in_universe
                tldf["universe_id"] = universe_ids
                targets = (
                    tldf.loc[tldf.in_universe]
                    .iloc[: self.n_systems_to_observe]
                    .reset_index(drop=True)
                )
                systems_to_observe = targets.universe_id.astype(int).tolist()
        elif "distance" in self.filters:
            dists = np.array(
                [
                    (system.star.dist.value, i)
                    for i, system in enumerate(universe.systems)
                ]
            )
            sorted_dists = dists[dists[:, 0].argsort()]
            sorted_ids = sorted_dists[: self.n_systems_to_observe, 1]
            systems_to_observe = [int(syst_id) for syst_id in sorted_ids]
        else:
            systems_to_observe = [i for i in range(self.n_systems_to_observe + 1)]
        return systems_to_observe

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
