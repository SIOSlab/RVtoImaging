import bz2
import json
import pickle
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

with open(".plot_config.json", "r") as f:
    config = json.load(f)

local = config["local"]
remote = config["remote"]

sim_script = "../test.json"
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

local_name = f"{local}_df.p"
remote_name = f"{remote}_df.p"


# Load the EXOSIMS script being used
with open(sim_script, "r") as f:
    script = json.load(f)

# Get all cached universes
_glob = Path("../.cache/").glob("universe*")
all_universes = [x for x in _glob if x.is_dir()]

relevant_params = {
    "missionStart": script["missionStart"],
    "commonSystemInclinations": script["commonSystemInclinations"],
    "PlanetPhysicalModel": script["modules"]["PlanetPhysicalModel"],
    "PlanetPopulation": script["modules"]["PlanetPopulation"],
    "SimulatedUniverse": script["modules"]["SimulatedUniverse"],
}
if not Path(local_name).exists():
    es_universes = []
    all_info = []
    for un in tqdm(all_universes, desc="Universe", position=0, leave=True):
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
                for star in tqdm(stars, desc="Stars", position=1, leave=False):
                    surveys = [x for x in Path(star).glob("*") if x.is_dir()]
                    for survey in surveys:
                        _obs_spec = Path(survey, "obs_spec.json")
                        _obs_data = Path(survey, "rv.csv")
                        fit_spec_path = Path(survey, "4_depth", "spec.json")
                        if fit_spec_path.exists():
                            # If the fit has been done then we catalog it
                            with open(_obs_spec, "r") as f:
                                obs_spec = json.load(f)
                            obs_data = pd.read_csv(_obs_data)
                            with open(fit_spec_path, "r") as f:
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
                                    # for obs_run, run_dict in obs_spec["obs_runs"]
                                    # .items():
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
                                    fit_info["observations"] = fit_spec.get(
                                        "observations"
                                    )
                                    fit_info["mcmc_success"] = fit_spec.get(
                                        "mcmc_success"
                                    )
                                    fit_info["fit_path"] = fit_spec_path
                                    csv_loc = Path(
                                        fit_spec_path.parent, "chains.csv.tar.bz2"
                                    )
                                    if "best_prob" in fit_spec.keys():
                                        fit_info["best_prob"] = fit_spec["best_prob"]
                                    else:
                                        if csv_loc.exists():
                                            try:
                                                with bz2.open(csv_loc, "rb") as f:
                                                    chains = pd.read_csv(f)
                                                fit_info["best_prob"] = chains.loc[
                                                    chains.lnprobability.idxmax(),
                                                    "lnprobability",
                                                ]
                                                # Store for later
                                                fit_spec["best_prob"] = chains.loc[
                                                    chains.lnprobability.idxmax(),
                                                    "lnprobability",
                                                ]
                                                with open(fit_spec_path, "w") as f:
                                                    json.dump(fit_spec, f)
                                            except EOFError:
                                                fit_info["best_prob"] = None

                                        else:
                                            fit_info["best_prob"] = None
                                    # Load the fit system
                                    system_location = Path(
                                        fit_spec_path.parent, "fitsystem.p"
                                    )
                                    if system_location.exists():
                                        with open(
                                            Path(fit_spec_path.parent, "fitsystem.p"),
                                            "rb",
                                        ) as f:
                                            system = pickle.load(f)
                                        fit_info["rms_vals"] = system.getpattr(
                                            "best_rms"
                                        ).value.tolist()
                                    all_info.append(fit_info)
    local_df = pd.DataFrame.from_dict(all_info, orient="columns")
    local_df.to_pickle(local_name)
else:
    local_df = pd.read_pickle(local_name)

if Path(remote_name).exists():
    remote_df = pd.read_pickle(remote_name)
    df = pd.concat([remote_df, local_df]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 10))
cmap = plt.get_cmap("viridis")
universes = np.unique(df.universe)
colors = [cmap(val) for val in np.linspace(0, 1, len(universes))]
# for dataset in datasets:
#     bottom_val = 0
#     for i, universe_number in enumerate(universes):
#         universe_color = colors[i]
#         rel_df = df.loc[(df.universe == universe_number) & (df.rv_dataset == dataset)]
#         n_converged = len(rel_df.loc[rel_df.mcmc_converged])
#         n_failed = len(rel_df) - n_converged
#         p = ax.bar(
#             dataset,
#             n_failed,
#             0.5,
#             label=universe_number,
#             bottom=bottom_val,
#             color=universe_color,
#             alpha=0.25,
#         )
#         p = ax.bar(
#             dataset,
#             n_converged,
#             0.5,
#             label=universe_number,
#             bottom=bottom_val + n_failed,
#             color=universe_color,
#         )
#         bottom_val += len(rel_df)
# legend_elements = [
#     mpl.patches.Patch(color=colors[i], label=f"{i}") for i in range(len(universes))
# ]
colors = [cmap(val) for val in np.linspace(0, 1, len(datasets))]
for universe_number in universes:
    bottom_val = 0
    for i, dataset in enumerate(datasets):
        dataset_color = colors[i]
        rel_df = df.loc[(df.universe == universe_number) & (df.rv_dataset == dataset)]
        n_converged = len(rel_df.loc[rel_df.mcmc_converged])
        n_failed = len(rel_df) - n_converged
        p = ax.bar(
            universe_number,
            n_failed,
            0.5,
            label=dataset,
            bottom=bottom_val,
            color=dataset_color,
            alpha=0.25,
        )
        p = ax.bar(
            universe_number,
            n_converged,
            0.5,
            label=dataset,
            bottom=bottom_val + n_failed,
            color=dataset_color,
        )
        bottom_val += len(rel_df)
legend_elements = [
    mpl.patches.Patch(color=colors[i], label=f"{dataset}")
    for i, dataset in enumerate(datasets)
]
# legend_elements.append(mpl.patches.Patch(color="k", alpha=1, label="MCMC converged"))
# legend_elements.append(mpl.patches.Patch(color="k", alpha=0.25, label="No fit"))
# mpl.lines.Line2D([0], [0], color = 'k', linewidth=lw, ls='dashed', label='Optimistic')
fig.subplots_adjust(top=0.9)
ax.set_ylabel("Number of systems")
ax.set_xticks(np.arange(universes.min(), universes.max() + 1, 1))
ax.legend(
    handles=legend_elements,
    ncols=int(len(universes) / 3) + 1,
    bbox_to_anchor=(0.5, 1),
    loc="lower center",
)
fig.savefig("progress.png", dpi=300)

# Analysis
filtered_systems = []
n_filtered_planets = 0
n_kept_planets = []
best_sigmas = []
for i, row in df.iterrows():
    if row.mcmc_converged:
        # Load the fit system
        system_location = Path(row.fit_path.parent, "fitsystem.p")
        if system_location.exists():
            with open(system_location, "rb") as f:
                system = pickle.load(f)
            filtered_systems.append(system)

            # Testing the filtering
            true_system = system.true_system
            planets_to_keep = []
            for planet in system.planets:
                per_Kerr = np.abs((true_system.getpattr("K") - planet.K) / planet.K)
                per_Terr = np.abs((true_system.getpattr("T") - planet.T) / planet.T)
                summed_error = per_Kerr + per_Terr

                # # Normalize the differences by the range of the system values
                # Knorm = np.abs(Kerr / (np.ptp(true_system.getpattr("K"))))
                # Tnorm = np.abs(Terr / (np.ptp(true_system.getpattr("T"))))

                closest_planet = np.argmin(summed_error)
                planet.closest_planet = closest_planet
                planet.closest_planet_err = summed_error[closest_planet].value
                if planet.closest_planet_err < 0.05:
                    planets_to_keep.append(planet)
                else:
                    n_filtered_planets += 1

                # breakpoint()
            system.filtered_planets = planets_to_keep
            system.rv_error = row.best_sigma
            n_kept_planets.append(len(planets_to_keep))
            best_sigmas.append(row.best_sigma)

kept_ratios = []
kept_errors = []
removed_ratios = []
removed_errors = []
for system in filtered_systems:
    true_system = system.true_system
    for planet in system.planets:
        match_planet = true_system.planets[planet.closest_planet]
        if planet in system.filtered_planets:
            kept_ratios.append(system.rv_error / match_planet.K.to(u.m / u.s).value)
            kept_errors.append(planet.closest_planet_err)
        else:
            match_planet = true_system.planets[planet.best_match]
            removed_ratios.append(system.rv_error / match_planet.K.to(u.m / u.s).value)
            removed_errors.append(planet.closest_planet_err)
            # kept_ratios.append(system.rv_error)

# plt.close()
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.scatter(
    kept_ratios, kept_errors, s=5, label="Considered candidated", color=cmap(0.99)
)
ax2.scatter(removed_ratios, removed_errors, s=5, label="Filtered out", color=cmap(0))
ax2.legend()
ax2.set_xlim([0, 2])
ax2.set_ylim([-0.1, 1])
ax2.set_xlabel(r"Ratio ($\frac{\sigma}{K}$)")
ax2.set_ylabel("Summed percent error on K and T compared to closest planet")
fig2.savefig("filtering.png")

fig3, ax3 = plt.subplots(figsize=(8, 8))
colors = [cmap(val) for val in np.linspace(0, 1, len(np.unique(best_sigmas)))]
for i, best_sigma in enumerate(np.unique(best_sigmas)):
    color = colors[i]
    sigma_inds = np.where(np.array(best_sigmas) == best_sigma)[0]
    n_kept_planets_sigma = np.array(n_kept_planets)[sigma_inds]
    p = ax3.bar(
        best_sigma,
        np.mean(n_kept_planets_sigma),
        0.1,
        # label=universe_number,
        # bottom=bottom_val,  # + n_failed,
        color=color,
    )
    # bottom_val += len(rel_df)
    # bottom_val += n_converged
ax3.set_xlabel("RV precision m/s")
ax3.set_ylabel("Average number of planets fit per system")
breakpoint()
plt.show()

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
