from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

import RVtools.utils as utils


def pop_3d(ax, pop, time_jd, color="r"):
    vectors = utils.calc_position_vectors(pop, Time([time_jd], format="jd"))
    x = (np.arctan((vectors["x"][0] * u.m) / pop.dist_to_star)).to(u.arcsec).value
    y = (np.arctan((vectors["y"][0] * u.m) / pop.dist_to_star)).to(u.arcsec).value
    z = (np.arctan((vectors["z"][0] * u.m) / pop.dist_to_star)).to(u.arcsec).value
    ax.scatter(x, y, z, alpha=0.75, s=0.01, color=color)
    return ax


def init_skyplot(fig, loc):
    ax = fig.add_subplot(loc, projection="aitoff")
    plt.grid(True)
    return ax


def sky_plot(ax, coord):
    gal = coord.galactic
    ax.scatter(gal.l.wrap_at("180d").radian, gal.b.radian)
    return ax


def image_plane(
    ax, pop, planet, time_jd, return_scatters=False, show_IWA=False, IWA_ang=None
):
    planet_pos = planet.calc_position_vectors(time_jd)
    ax.scatter(0, 0, s=250, zorder=2, color="black")
    pop_pos = pop.calc_position_vectors(time_jd)

    # Add the planets at their current location
    p_scatter = ax.scatter(
        np.arctan(planet_pos[0].to(u.AU) / planet.dist_to_star).to(u.arcsec).value,
        np.arctan(planet_pos[1].to(u.AU) / planet.dist_to_star).to(u.arcsec).value,
        s=20,
        label="Ground truth planet",
        zorder=3,
        marker="x",
        # color=pcolor,
    )
    pop_scatter = ax.scatter(
        np.arctan(pop_pos[0, :].to(u.AU) / pop.dist_to_star).to(u.arcsec).value,
        np.arctan(pop_pos[1, :].to(u.AU) / pop.dist_to_star).to(u.arcsec).value,
        s=0.1,
        alpha=0.5,
        label="Orbit fit",
        # color=ecolor,
    )

    separation_lim = 0.15
    # Now set plot limits
    ax.set_xlim([-separation_lim, separation_lim])
    ax.set_ylim([-separation_lim, separation_lim])
    ax.set_xlabel('Apparent separation in x (")')
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Apparent separation in y (")')

    if show_IWA:
        IWA_patch = mpatches.Circle(
            (0, 0),
            IWA_ang.to(u.arcsec).value,
            facecolor="grey",
            edgecolor="black",
            alpha=0.5,
            zorder=5,
        )
        ax.add_patch(IWA_patch)
        if ax.get_subplotspec().is_first_col():
            ax.annotate(
                "IWA",
                xy=(0, 0),
                xytext=(0, IWA_ang.to(u.arcsec).value * 1.125),
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="<-"),
                zorder=10,
            )
    if return_scatters:
        return ax, (p_scatter, pop_scatter)
    else:
        return ax


def base_plot(ax, times, val, final_ind=-1):
    cumulative_times = times.value - times[0].value
    ax.set_xlim(
        [
            times[0].value - times[0].value,
            times[-1].value - times[0].value,
        ]
    )
    ax.set_xlabel("Time (yr)")
    ax.plot(
        cumulative_times[:final_ind],
        val[:final_ind],
        # color=ecolor,
    )
    return ax


def entropy_plot(
    ax,
    times,
    entropy,
    entropy_bounds,
    final_ind=-1,
    add_hatching=False,
    hatching_info=None,
):
    ax = base_plot(ax, times, entropy, final_ind=final_ind)
    ax.set_yscale("symlog")
    ax.set_ylim([entropy_bounds[0], entropy_bounds[1]])
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Entropy (arbitrary units)")

    # Add hatching for true detectability
    if add_hatching:
        planet, IWA, OWA = hatching_info
        ax = add_true_planet_hatching(ax, times, planet, IWA, OWA, -1)
    return ax


def distance_plot(
    ax, times, dists, dists_bounds, final_ind=-1, hatching_info=None, add_hatching=False
):
    ax = base_plot(ax, times, dists, final_ind=final_ind)
    ax.set_ylim([dists_bounds[0], dists_bounds[1]])
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Mean distance (AU)")
    # Add hatching for true detectability
    if add_hatching:
        planet, IWA, OWA = hatching_info
        ax, det_sq, not_det_sq = add_true_planet_hatching(
            ax, times, planet, IWA, OWA, -1, return_patches=True
        )
        return ax, det_sq, not_det_sq
    return ax


def pdet_plot(ax, times, pdet, final_ind=-1, hatching_info=None, add_hatching=False):
    ax = base_plot(ax, times, pdet, final_ind=final_ind)
    ax.set_ylim(-0.05, 1.05)
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Probability of detection")
    ax.set_xlabel("Time (yr)")

    # Add hatching for true detectability
    if add_hatching:
        planet, IWA, OWA = hatching_info
        ax, det_sq, not_det_sq = add_true_planet_hatching(
            ax, times, planet, IWA, OWA, -1, return_patches=True
        )
        return ax, det_sq, not_det_sq

    else:
        return ax


def add_true_planet_hatching(
    ax, times, planet, IWA, OWA, dMag0, final_ind, return_patches=False
):
    cumulative_times = times.value - times[0].value
    true_pdet = []
    for i, time in enumerate(times[:final_ind]):
        time_jd = Time(Time(time, format="decimalyear").jd, format="jd")
        planet_alpha, planet_dMag = planet.prop_for_imaging(time_jd)
        true_pdet.append(
            (planet_dMag < dMag0)
            & (OWA.to(u.arcsec).value > planet_alpha.to(u.arcsec).value)
            & (planet_alpha.to(u.arcsec).value > IWA.to(u.arcsec).value)
        )
    change_inds = np.where(np.roll(true_pdet, 1) != true_pdet)[0]
    inds_to_plot = np.concatenate(([0], change_inds, [len(true_pdet) - 1]))
    ymin, ymax = ax.get_ylim()
    box_height = ymax - ymin
    plt.rcParams["hatch.linewidth"] = 0.5
    for i, ind in enumerate(inds_to_plot):
        current_pdet = true_pdet[ind]
        if i + 1 == len(inds_to_plot):
            continue
        if current_pdet == 1:
            start_time = cumulative_times[ind]
            width = cumulative_times[inds_to_plot[i + 1]] - cumulative_times[ind]
            det_sq1 = mpl.patches.Rectangle(
                (start_time, ymin),
                width=width,
                height=box_height,
                zorder=2,
                edgecolor="black",
                fill=False,
            )
            ax.add_patch(det_sq1)
            det_sq = mpl.patches.Rectangle(
                (start_time, ymin),
                width=width,
                height=box_height,
                hatch=r"\\",
                zorder=1,
                alpha=0.5,
                edgecolor="black",
                label="Ground truth planet detectable",
                fill=False,
            )
            ax.add_patch(det_sq)
        else:
            start_time = cumulative_times[ind]
            width = cumulative_times[inds_to_plot[i + 1]] - cumulative_times[ind]
            not_det_sq1 = mpl.patches.Rectangle(
                (start_time, ymin),
                width=width,
                height=box_height,
                zorder=2,
                edgecolor="black",
                fill=False,
            )
            ax.add_patch(not_det_sq1)
            not_det_sq = mpl.patches.Rectangle(
                (start_time, ymin),
                width=width,
                height=box_height,
                hatch="||",
                zorder=1,
                alpha=0.5,
                edgecolor="black",
                label="Ground truth planet not detectable",
                fill=False,
            )
            ax.add_patch(not_det_sq)
    if return_patches:
        return ax, det_sq, not_det_sq
    else:
        return ax


def fig1(pop, planet, entropy, all_times, IWA, OWA):
    """
    Figure that demonstrates how entropy is related to the spread of the constructed
    orbits and probability of detection
    """
    font = {"size": 18}
    plt.rc("font", **font)
    plt.rcParams["lines.linewidth"] = 4
    # inds = [10, 120, 182, 253, -1]
    inds = [10, 120, 182, -1]
    plot_times = [
        Time(Time(all_times[ind], format="decimalyear").jd, format="jd") for ind in inds
    ]
    pdets = pop.percent_detectable

    fig1 = plt.figure(figsize=[24, 20])
    subfigs = fig1.subfigures(3, 1, height_ratios=[2, 2, 2])

    # Image planes
    im_axes = subfigs[0].subplots(1, len(plot_times), sharey=True)
    for (im_ax, time) in zip(im_axes, plot_times):
        im_ax, scatters = image_plane(
            im_ax, pop, planet, time, show_IWA=True, return_scatters=True
        )
        im_ax.set_title(f"Time = {time.decimalyear-1:.2f} yr")
        if im_ax.get_subplotspec().is_first_col():
            im_ax_leg = im_ax.legend(
                handles=scatters,
                loc="lower center",
                # bbox_to_anchor=(0.5, 0.87),
            )
            for legend_handle in im_ax_leg.legendHandles:
                legend_handle.set_sizes(sizes=[25])

    # im_ax_leg = subfigs[1].legend(
    #     handles=scatters,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 0.87),
    # )

    # Entropy plot
    entropy_axes = subfigs[1].subplots(1, len(plot_times), sharey=True)
    for (entropy_ax, final_ind) in zip(entropy_axes, inds):
        entropy_ax = entropy_plot(
            entropy_ax,
            all_times,
            entropy,
            final_ind=final_ind,
            add_hatching=True,
            hatching_info=(planet, IWA, OWA),
        )

    # pdet plot
    pdet_axes = subfigs[2].subplots(1, len(plot_times), sharey=True)
    for (pdet_ax, final_ind) in zip(pdet_axes, inds):
        pdet_ax, det_sq, not_det_sq = pdet_plot(
            pdet_ax,
            all_times,
            pdets,
            final_ind=final_ind,
            add_hatching=True,
            hatching_info=(planet, IWA, OWA),
        )
    subfigs[2].legend(
        handles=[det_sq, not_det_sq], bbox_to_anchor=(0.5, 1.03), loc="upper center"
    )
    fig1.savefig(Path("time_progression.png"), dpi=150)
