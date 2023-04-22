import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sim_script = "test.json"
seed_range = range(30, 45)
datasets = ["conservative", "baseline", "optimistic"]
rv_terms = [
    "sigma_rv",
    "rv",
    "instrument",
    "photon",
    "pmode_oscillation",
    "granulation",
    "magnetic",
]

# Load the EXOSIMS script being used
with open(sim_script, "r") as f:
    script = json.load(f)

# Get all cached universes
_glob = Path(".cache/").glob("universe*")
all_universes = [x for x in _glob if x.is_dir()]

relevant_params = {
    "missionStart": script["missionStart"],
    "commonSystemInclinations": script["commonSystemInclinations"],
    "PlanetPhysicalModel": script["modules"]["PlanetPhysicalModel"],
    "PlanetPopulation": script["modules"]["PlanetPopulation"],
    "SimulatedUniverse": script["modules"]["SimulatedUniverse"],
}
es_universes = []
all_info = []
for un in all_universes:
    if Path(un, "spec.json").exists():
        with open(Path(un, "spec.json"), "r") as f:
            _spec = json.load(f)
        # Filter out exovista universes
        if _spec["universe_type"] == "exosims":
            es_universes.append(un)
        else:
            continue

        # Filter out EXOSIMS universes with the wrong parameters
        valid_universe = True
        for param, val in relevant_params.items():
            if param in _spec.keys():
                if _spec[param] != val:
                    valid_universe = False
                    break
            else:
                valid_universe = False
                break
        if valid_universe:
            # Go through and check the fitting status of the stars
            star_glob = Path(un).glob("HIP_*")
            stars = [x for x in star_glob if x.is_dir()]
            for star in stars:
                surveys = [x for x in Path(star).glob("*") if x.is_dir()]
                for survey in surveys:
                    _obs_spec = Path(survey, "obs_spec.json")
                    _obs_data = Path(survey, "rv.csv")
                    _fit_spec = Path(survey, "4_depth", "spec.json")
                    if _fit_spec.exists():
                        # If the fit has been done then we catalog it
                        with open(_obs_spec, "r") as f:
                            obs_spec = json.load(f)
                        obs_data = pd.read_csv(_obs_data)
                        with open(_fit_spec, "r") as f:
                            fit_spec = json.load(f)
                        info = []
                        if dataset_name := obs_spec.get("rv_dataset"):
                            if dataset_name in datasets:
                                fit_info = {}
                                fit_info["universe"] = _spec["seed"]
                                fit_info["star"] = star.parts[-1]
                                fit_info["rv_dataset"] = dataset_name

                                # Get the best precision
                                # sigmas = []
                                # for obs_run, run_dict in obs_spec["obs_runs"].items():
                                #     run_sigma_vals = []
                                #     for rv_term in rv_terms:
                                #         if run_dict.get(rv_term):
                                #             run_sigma_vals.append(run_dict[rv_term])
                                #     sigmas.append(
                                #         np.sqrt(np.mean(np.square(run_sigma_vals)))
                                #     )
                                fit_info["best_sigma"] = min(obs_data.errvel)

                                # Fit information
                                fit_info["planets_fitted"] = fit_spec.get(
                                    "planets_fitted"
                                )
                                fit_info["observational_baseline"] = fit_spec.get(
                                    "observational_baseline"
                                )
                                fit_info["mcmc_converged"] = fit_spec.get(
                                    "mcmc_converged"
                                )
                                fit_info["observations"] = fit_spec.get("observations")
                                fit_info["mcmc_success"] = fit_spec.get("mcmc_success")
                                fit_info["fit_path"] = _fit_spec

                                # Load the fit system
                                system_location = Path(_fit_spec.parent, "fitsystem.p")
                                if system_location.exists():
                                    with open(
                                        Path(_fit_spec.parent, "fitsystem.p"), "rb"
                                    ) as f:
                                        system = pickle.load(f)
                                    fit_info["rms_vals"] = system.getpattr(
                                        "best_rms"
                                    ).value.tolist()
                                all_info.append(fit_info)
df = pd.DataFrame.from_dict(all_info, orient="columns")
fig, ax = plt.subplots(figsize=(10, 10))
cmap = plt.get_cmap("viridis")
universes = np.unique(df.universe)
colors = [cmap(val) for val in np.linspace(0, 1, len(universes))]

for dataset in datasets:
    bottom_val = 0
    for i, universe_number in enumerate(universes):
        universe_color = colors[i]
        rel_df = df.loc[(df.universe == universe_number) & (df.rv_dataset == dataset)]
        n_converged = len(rel_df.loc[rel_df.mcmc_converged])
        n_failed = len(rel_df) - n_converged
        p = ax.bar(
            dataset,
            n_failed,
            0.5,
            label=universe_number,
            bottom=bottom_val,
            color=universe_color,
            alpha=0.25,
        )
        p = ax.bar(
            dataset,
            n_converged,
            0.5,
            label=universe_number,
            bottom=bottom_val + n_failed,
            color=universe_color,
        )
        bottom_val += len(rel_df)
legend_elements = [
    mpl.patches.Patch(color=colors[i], label=f"{universes[i]}")
    for i in range(len(universes))
]
legend_elements.append(mpl.patches.Patch(color="k", alpha=1, label="converged"))
legend_elements.append(mpl.patches.Patch(color="k", alpha=0.25, label="mcmc failed"))
# mpl.lines.Line2D([0], [0], color = 'k', linewidth=lw, ls='dashed', label='Optimistic')
fig.subplots_adjust(top=0.75)
ax.legend(
    handles=legend_elements,
    ncols=len(universes) - 1,
    bbox_to_anchor=(0.5, 1),
    loc="lower center",
)
fig.savefig("progress.png", dpi=300)
# breakpoint()
# hip.Experiment.from_iterable(df).display()
# cmap = plt.get_cmap("viridis")
# universes = np.unique(df.universe)

# fig, ax = plt.subplots()
# for dataset in datasets:
#     bottom_val = 0
#     for i, universe_number in enumerate(universes):
#         universe_color = cmap(i / len(universes))
#         rel_df = df.loc[(df.universe == universe_number) & (df.rv_dataset == dataset)]
#         if len(rel_df):
#             breakpoint()
