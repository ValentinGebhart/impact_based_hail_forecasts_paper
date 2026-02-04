# modules

# modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
import pandas as pd
import geopandas as gpd
from scipy import stats

from climada import CONFIG

from utils import ensemble_range

# parameters for plotting
cantons_extent_x = [-110000, 75000]
cantons_extent_y = [-60000, 120000]

CH_extent_x = [-180000, 180000]
CH_extent_y = [-120000, 130000]

MAP_FONTSIZE = 12
MAP_EXTENT = [5.5, 11.2, 45.4, 48.1]
ch_shp_path = f"{CONFIG.local_data.data_dir}/ch_shapefile"


###################################
###### cantonal forecast plot #####
###################################


def plot_cantonal_imp(
    imp,
    exp_n_build,
    day,
    init,
    cmap,
    data_source="",
    return_ax=False,
    bin_params=None,
    show_init_valid=True,
    figsize_scale=1.0,
):
    """plot of forecast-modeled impact per canton using pie plots

    Args:
        imp (impact object): impact object with different ensemble members for given day
        day (dt.datetime): date for plot
        init (dt.datetime): date for initialization of forecast
        cmap (matplotlib.colors.ListedColormap): colormap for pie plots
        data_source (str, optional): forecast data source. Defaults to ''.
        return_ax (bool, optional): to return fig with axis . Defaults to False.
        bin_params (tuple, optional): parameters (thresh_low, thresh_high, n_bins) for bins for pie plots_description_. Defaults to None.
                                        thresh_low: lower bound of lowest bin
                                        thresh_high: upper bound of highest bin
                                        n_bins: number of bins
                                    Defaults to (.0005, .05, 5)

    Returns:
        fig: figure of forecast-modeled impact per canton using pie plots
    """

    # gdf with impacts and exposures
    coord_df = pd.DataFrame(imp.coord_exp, columns=["latitude", "longitude"])
    gdf = gpd.GeoDataFrame(
        coord_df,
        geometry=gpd.points_from_xy(
            imp.coord_exp[:, 1], imp.coord_exp[:, 0], crs="EPSG:4326"
        ),
    )
    for event_id in imp.event_id:
        gdf[event_id] = imp.select(event_ids=[event_id]).imp_mat.toarray().T[:, 0]
    gdf["exp_total"] = exp_n_build.gdf["value"]

    # load canton data
    cantons = gpd.read_file("%s/swissTLMRegio_KANTONSGEBIET_LV95.shp" % ch_shp_path)
    cantons = cantons.to_crs(epsg=4326)
    aggregated_gdf = cantons.dissolve(by="NAME", as_index=False)
    aggregated_gdf = aggregated_gdf.loc[aggregated_gdf.ICC == "CH", :].reset_index(
        drop=True
    )  # select only Swiss admin0 regions (i.e. Cantons)

    # Perform spatial join
    joined_gdf = gpd.sjoin(aggregated_gdf, gdf, how="inner", predicate="intersects")

    # Sum up the impacts column for each polygon in aggregated_gdf
    canton_impact = joined_gdf.groupby("NAME")[["exp_total", *imp.event_id]].sum()
    canton_impact["centerofmass"] = [
        np.array(aggregated_gdf.to_crs(epsg=2056)["geometry"][k].centroid.coords[0])
        for k in range(aggregated_gdf.shape[0])
    ]

    # change centerofmass for better visualization
    canton_impact["centerofmass"]["Basel-Landschaft"] += np.array([10000, 0])
    canton_impact["centerofmass"]["Obwalden"] += np.array([-5000, 0])
    canton_impact["centerofmass"]["Nidwalden"] += np.array([0, 2000])
    canton_impact["centerofmass"]["Schwyz"] += np.array([1000, 0])
    canton_impact["centerofmass"]["St. Gallen"] += np.array([-12000, -1000])
    canton_impact["centerofmass"]["Appenzell Ausserrhoden"] += np.array([-5000, 5000])
    canton_impact["centerofmass"]["Appenzell Innerrhoden"] += np.array([5000, -5000])

    # create plot for cantonal impacts
    if show_init_valid:
        valid_time = (day + pd.Timedelta(6, "h"), day + pd.Timedelta(24 + 6, "h"))
        init_time = init
    else:
        valid_time, init_time = "", ""

    fig, ax, cbax = plot_standard_map(
        data_source=data_source,
        valid_time=valid_time,
        init_time=init_time,
        figsize=(10 * figsize_scale, 6.25 * figsize_scale),
    )
    plot_canton(
        ax, canton="all", edgecolor="gray", facecolor="none", linewidth=0.5, zorder=2
    )
    plot_country(
        ax, country="all", edgecolor="grey", facecolor="none", linewidth=1, zorder=4
    )
    plot_country(
        ax,
        country=["AT", "LI", "DE", "IT", "FR"],
        edgecolor="none",
        facecolor="lightgrey",
        linewidth=1,
        zorder=3,
        alpha=1,
    )
    plot_lakes(ax, edgecolor="none", facecolor="lightblue", zorder=5)
    plot_canton(
        ax,
        canton=["Zürich", "Bern", "Luzern", "Aargau"],
        edgecolor="black",
        facecolor="none",
        linewidth=1.0,
        zorder=3,
    )

    # if only four cantons, we zoom in
    ax.set_xlim(*CH_extent_x)
    ax.set_ylim(*CH_extent_y)

    # add small plots
    inside_plots = {}
    for canton in canton_impact.T:
        x_loc, y_loc = canton_impact.loc[canton, "centerofmass"]
        box_width, box_height = (25000, 25000)
        inside_plots[canton] = ax.inset_axes(
            [x_loc - box_width / 2, y_loc - box_height / 2, box_width, box_height],
            transform=ccrs.epsg(2056)._as_mpl_transform(ax),
        )

        # insert pie charts
        if bin_params == None:
            thresh_low, thresh_high, n_bins = (0.0005, 0.05, 5)
        else:
            thresh_low, thresh_high, n_bins = bin_params
        bins = np.geomspace(thresh_low, thresh_high, n_bins)
        bins_with_borders = np.geomspace(
            thresh_low
            / 10 ** ((np.log10(thresh_high) - np.log10(thresh_low)) / (n_bins - 1)),
            thresh_high
            * 10 ** ((np.log10(thresh_high) - np.log10(thresh_low)) / (n_bins - 1)),
            n_bins + 2,
        )
        dmgs_rel = sorted(
            canton_impact.loc[canton][[*imp.event_id]].values
            / canton_impact.loc[canton, "exp_total"]
        )
        dmgs_rel = np.minimum(
            np.maximum(dmgs_rel, bins_with_borders[0] * 1.01),
            bins_with_borders[-1] * 0.99,
        )
        dmgs_binned_log = np.log10(
            bins_with_borders[pd.cut(dmgs_rel, bins_with_borders).codes]
        )
        colors = cmap(
            (dmgs_binned_log - np.log10(bins_with_borders[0]))
            / (np.log10(bins_with_borders[-2]) - np.log10(bins_with_borders[0]))
        )
        colors = [
            color if color[0] < 0.99714 else [1.0, 1.0, 1.0, 1.0] for color in colors
        ]  # remove line if no white
        inside_plots[canton].pie(
            np.ones(len(imp.at_event)),
            labels=None,
            colors=colors,
            startangle=90,
            wedgeprops={"linewidth": 0.1, "edgecolor": "black", "alpha": 1.0},
        )

    # colorbar
    bounds = (np.log10(bins) - np.log10(bins[0])) / (
        np.log10(bins[-2]) - np.log10(bins[0])
    )
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend="both")
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax, orientation="vertical"
    )
    cbax.set_yticklabels([f"{np.round(bin*100,2)}%" for bin in bins])
    cbar.set_label("Percentage of damaged residential buildings per canton")

    if return_ax:
        return fig, ax
    return fig


def plot_imp_hist(
    imp,
    imp_Q05,
    imp_Q95,
    day,
    init,
    bin_params=None,
    colors=None,
    data_source="",
    return_ax=False,
    show_init_valid=True,
    figsize_scale=1.0,
):
    """plot of forecast-modeled impact histogram for Switzerland

    Args:
        imp (impact object): impact object from best fit impact function with different ensemble members
        imp_Q05 (impact object): impact object from 5% quantile impact function with different ensemble members
        imp_Q95 (impact object): impact object from 95% quantile impact function with different ensemble members
        day (dt.datetime): date for plot
        init (dt.datetime): date for initialization of forecast
        bin_params (tuple, optional): parameters (thresh_low, thresh_high, n_bins) for bins for pie plots. Defaults to (1, 100000, 11).
                                        thresh_low: lower bound of lowest bin
                                        thresh_high: upper bound of highest bin
                                        n_bins: number of bins
        colors (tupe): colors (col_best_fit, col_quantiles) for plotting best fit impacts and impacts including 5% and 95%
                        quantiles. Defaults to ('orangered', 'gray').
        data_source (str, optional): forecast data source. Defaults to ''.
        return_ax (bool, optional): to return fig with axis . Defaults to False.
        figsize_scale (float, optional): scale factor for figure size. Defaults to 1.0.

    Returns:
        fig: figure of forecast-modeled impact histogram for Switzerland
    """

    # cut damages to bins
    if bin_params == None:
        bin_params = (1, 100000, 11)
    bins = np.geomspace(*bin_params)
    dmgs_mod = imp.at_event.clip(min=bins[0], max=bins[-1])
    dmgs_mod_Q05 = imp_Q05.at_event.clip(min=bins[0], max=bins[-1])
    dmgs_mod_Q95 = imp_Q95.at_event.clip(min=bins[0], max=bins[-1])
    dmgs_combined = np.concatenate((dmgs_mod, dmgs_mod_Q05, dmgs_mod_Q95), axis=None)

    # fit damages
    log_dmgs = np.log10(dmgs_mod)
    log_dmgs_combined = np.log10(dmgs_combined)
    if any(log_dmgs > np.log10(bins[0]) + 0.00001):
        kernel = stats.gaussian_kde(log_dmgs)
    else:
        kernel = stats.gaussian_kde(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
        )
    if any(log_dmgs_combined > np.log10(bins[0]) + 0.00001):
        kernel_all = stats.gaussian_kde(log_dmgs_combined)
    else:
        kernel_all = stats.gaussian_kde(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
        )

    # compute quantiles
    dmgs_combined = np.concatenate((dmgs_mod, dmgs_mod_Q05, dmgs_mod_Q95), axis=None)
    quantiles_all = {q: np.quantile(dmgs_combined, q) for q in [0.05, 0.25, 0.75, 0.95]}

    # initialize plot
    if show_init_valid:
        valid_time = (day + pd.Timedelta(6, "h"), day + pd.Timedelta(24 + 6, "h"))
        init_time = init
    else:
        valid_time, init_time = "", ""
    # fig, ax, cbax = plot_standard_map_n2o(data_source=data_source, valid_time=valid_time, init_time = init_time, prj = None)
    fig, ax, cbax = plot_standard_map(
        data_source=data_source,
        valid_time=valid_time,
        init_time=init_time,
        prj=None,
        figsize=(10 * figsize_scale, 6.25 * figsize_scale),
    )

    cbax.remove()  # hide cbax
    ax.set_position([0, 0.2, 1.1, 0.8])
    ax.set_xscale("log")

    # plot histograms

    alpha_hist = (0.9, 0.2)
    xlims = (bin_params[0], bin_params[1])
    if colors == None:
        colors = (np.array([230, 50, 30, 255]) / 255, "gray")
    _, bins, _ = ax.hist(
        dmgs_mod.clip(min=bins[0], max=bins[-1]),
        bins=bins,
        edgecolor="black",
        alpha=alpha_hist[0],
        color=colors[0],
    )
    _, bins, rects = ax.hist(
        dmgs_combined.clip(min=bins[0], max=bins[-1]),
        bins=bins,
        edgecolor="black",
        alpha=alpha_hist[1],
        color=colors[1],
    )
    for r in rects:
        r.set_height(r.get_height() / 3)
    # patches[0].set_facecolor('lightgreen')

    # plot fits
    outline = mpl.patheffects.withStroke(linewidth=6, foreground="black")
    normalization = (
        len(dmgs_mod)
        * (np.log10(bin_params[1]) - np.log10(bin_params[0]))
        / (bin_params[2] - 1)
    )
    ax.plot(
        np.geomspace(bins[0], bins[-1], 1000),
        normalization
        * kernel_all(np.linspace(np.log10(bins[0]), np.log10(bins[-1]), 1000)),
        color=colors[1],
        linewidth=4,
        path_effects=[outline],
    )
    ax.plot(
        np.geomspace(bins[0], bins[-1], 1000),
        normalization
        * kernel(np.linspace(np.log10(bins[0]), np.log10(bins[-1]), 1000)),
        color=colors[0],
        linewidth=4,
        path_effects=[outline],
    )

    # plot scatter damgs
    ax_scatter = fig.add_axes([0, 0.1, 1.038, 0.1])
    ax_scatter.set_xscale("log")
    ax_scatter.scatter(dmgs_mod, [1.0] * len(dmgs_mod), s=30, c=colors[0], alpha=0.9)
    ax_scatter.scatter(
        dmgs_mod_Q05, [0.95] * len(dmgs_mod), s=25, alpha=0.6, c=colors[1]
    )
    ax_scatter.scatter(
        dmgs_mod_Q95, [1.05] * len(dmgs_mod), s=25, alpha=0.6, c=colors[1]
    )
    ax_scatter.set_ylim(0.9, 1.1)

    # plot confidence intervals
    ax_intervals = fig.add_axes([0, 0.0, 1.038, 0.1])
    ax_intervals.set_xscale("log")
    for anchor, p_min, p_max, col, label_interval in zip(
        [0.9, 1.0],
        [0.05, 0.25],
        [0.95, 0.75],
        ["lightgray", "darkgray"],
        ["90% of members", "50% of members"],
    ):
        ax_intervals.add_patch(
            mpl.patches.Rectangle(
                (quantiles_all[p_min], anchor),
                quantiles_all[p_max] - quantiles_all[p_min],
                0.1,
                facecolor=col,
            )
        )
        ax_intervals.annotate(
            label_interval,
            (0.95 * xlims[1], anchor + 0.05),
            color="k",
            ha="right",
            va="center",
        )
    ax_intervals.set_ylim(0.9, 1.1)

    # set y ticks
    ax.set_ylim(0, 11)
    ax.set_yticks(np.array([0.2 * i for i in range(6)]) * len(dmgs_mod))
    ax.set_yticklabels([f"{20*i}%" for i in range(6)])
    ax.set_ylabel("probability")
    ax_scatter.set_yticks([])
    ax_intervals.set_yticks([])

    # set x ticks
    ax.set_xlim(*xlims)
    ax_scatter.set_xlim(*xlims)
    ax_intervals.set_xlim(*xlims)
    ax.set_xticklabels([])
    ax_scatter.set_xticklabels([])
    width = int(np.log10(bins[-1]) - np.log10(bins[0]))
    ax_intervals.set_xticks(
        np.logspace(np.log10(bins[0]), np.log10(bins[0]) + width, width + 1)
    )
    xticklabels = [str(int(tick)) for tick in ax_intervals.get_xticks()]
    xticklabels[0] = "no \n damage"
    ax_intervals.set_xticklabels(xticklabels)
    ax_intervals.set_xlabel(f"number of damaged residential buildings in Switzerland")

    # add a legend
    color_patch = mpl.patches.Patch(color=colors[0], label="best fit impact function")
    gray_patch = mpl.patches.Patch(
        color=colors[1], label="5% and 95% quantiles impact functions"
    )
    ax.legend(handles=[color_patch, gray_patch], loc="upper right")

    if return_ax:
        return fig, ax
    return fig


# ensemble ranges of largest forecasts
def plot_ranges_largest_forecasts(
    estimated_large, dates_large, OOM_threhsold, threshold_prediction, reported_large
):

    q0595 = np.array(
        ensemble_range(
            np.round(estimated_large, decimals=0),
            0.05,
            0.95,
            threshold_values=threshold_prediction,
        )
    ).T
    q00100 = np.array(
        ensemble_range(
            np.round(estimated_large, decimals=0),
            0.00,
            1.0,
            threshold_values=threshold_prediction,
        )
    ).T
    q2575 = np.array(
        ensemble_range(
            np.round(estimated_large, decimals=0),
            0.25,
            0.75,
            threshold_values=threshold_prediction,
        )
    ).T

    mins, maxs = q00100.T
    q05, q95 = q0595.T
    q25, q75 = q2575.T

    y_pos = range(len(mins))

    fig = plt.figure(figsize=(8, 14))

    # Draw line for each min-max range
    for y, mn, mx in zip(y_pos, mins, maxs):
        plt.hlines(y, mn, mx, color="LightGray", alpha=0.5, linewidth=5)
    for y, mn, mx in zip(y_pos, q05, q95):
        plt.hlines(y, mn, mx, color="Gray", alpha=0.7, linewidth=5)
    for y, mn, mx in zip(y_pos, q25, q75):
        plt.hlines(y, mn, mx, color="k", linewidth=5)

    plt.vlines(np.log10(OOM_threhsold), 0, len(mins), linewidth=1, linestyles="dotted")

    if reported_large is not None:
        plt.scatter(
            np.log10(np.clip(reported_large, a_min=threshold_prediction, a_max=None)),
            y_pos,
            marker="|",
            linewidth=2,
            zorder=10,
            color="red",
        )

    plt.xlabel("Estimated number of damaged buildings")
    plt.yticks(y_pos, [str(date)[:10] for date in dates_large], fontsize=9)
    plt.xticks(
        [np.log10(0.3), 0, 1, 2, 3, 4], ["no \n damage", 1, 10, 100, 1000, 10000]
    )
    plt.ylim([-1, len(mins)])
    plt.grid(axis="y", alpha=0.3, which="both")

    return fig


# ensemble ranges of largest forecasts
def plot_ranges_largest_forecasts_ch_and_cantons(
    estimated_large,
    dates_large,
    OOM_threhsold,
    threshold_prediction,
    estimated_cantons,
    reported_cantons,
):
    quantiles_ch, qunatiles_4cantons = {}, {}
    for estimated, quantiles in zip(
        [estimated_large, estimated_cantons], [quantiles_ch, qunatiles_4cantons]
    ):
        for q, qmin, qmax in zip(
            ["q0595", "q00100", "q2575"], [0.05, 0.00, 0.25], [0.95, 1.0, 0.75]
        ):
            quantiles[q] = np.array(
                ensemble_range(
                    np.round(estimated, decimals=0),
                    qmin,
                    qmax,
                    threshold_values=threshold_prediction,
                )
            ).T

    fig = plt.figure(figsize=(6, 14))

    # Use GridSpec for fine control of spacing
    gs = fig.add_gridspec(1, 2, wspace=0)  # wspace=0 removes gap between panels

    # Create left and right panels with shared y-axis
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left, sharex=ax_left)

    for ax, quantiles in zip([ax_left, ax_right], [quantiles_ch, qunatiles_4cantons]):
        mins, maxs = quantiles["q00100"].T
        q05, q95 = quantiles["q0595"].T
        q25, q75 = quantiles["q2575"].T
        y_pos = range(len(mins))

        # Draw line for each min-max range
        for y, mn, mx in zip(y_pos, mins, maxs):
            ax.hlines(y, mn, mx, color="LightGray", alpha=0.5, linewidth=5)
        for y, mn, mx in zip(y_pos, q05, q95):
            ax.hlines(y, mn, mx, color="Gray", alpha=0.7, linewidth=5)
        for y, mn, mx in zip(y_pos, q25, q75):
            ax.hlines(y, mn, mx, color="k", linewidth=5)

    ax_left.vlines(
        np.log10(OOM_threhsold), 0, len(mins), linewidth=1, linestyles="dotted"
    )

    ax_right.scatter(
        np.log10(np.clip(reported_cantons, a_min=threshold_prediction, a_max=None)),
        y_pos,
        marker="|",
        linewidth=2,
        zorder=10,
        color="red",
    )

    ax_left.set_yticks(y_pos, [str(date)[:10] for date in dates_large], fontsize=9)
    ax_left.set_xticks(
        [np.log10(threshold_prediction), 0, 1, 2, 3, 4], ["<1", 1, 10, 100, 1000, "10k"]
    )
    ax_left.set_ylim([-1, len(mins)])
    for ax in [ax_left, ax_right]:
        ax.grid(axis="y", alpha=0.3, which="both")

    ax_left.set_title("All cantons")
    ax_right.set_title("Validation cantons")

    # Hide duplicate y-tick labels on the right panel
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Optional: tighten layout without gaps
    fig.subplots_adjust(wspace=0)
    fig.supxlabel("Estimated number of damaged buildings", fontsize=12, y=0.07)

    return fig


# colormap function
def get_cmap_killian(name="plasma", levels=None):

    if levels is None:
        levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    cmap = plt.cm.get_cmap(name, len(levels) + 1)
    colors = list(cmap(np.arange(len(levels) + 1)))
    cmap = mpl.colors.ListedColormap(colors[1:-1], "")
    cmap.set_over(colors[-1])
    return cmap


# empty plot for Switzerland
def create_empty_plot_CH(
    title="",
    four_cantons=False,
    four_cantons_style="solid",
    figsize=(10, 6.25),
    show_cantons=True,
):
    # create plot for cantonal impacts
    fig, ax, cbax = plot_standard_map(
        data_source=title,
        valid_time="",  # (day + pd.Timedelta(6, "h"), day + pd.Timedelta(24 + 6, "h")),
        init_time="",
        figsize=figsize,
    )
    if show_cantons:
        plot_canton(
            ax,
            canton="all",
            edgecolor="gray",
            facecolor="none",
            linewidth=0.5,
            zorder=2,
        )
    if four_cantons:
        plot_canton(
            ax,
            canton=["Zürich", "Bern", "Luzern", "Aargau"],
            edgecolor="black",
            facecolor="none",
            linewidth=1.0,
            zorder=2,
            linestyle=four_cantons_style,
        )

    plot_country(
        ax, country="all", edgecolor="grey", facecolor="none", linewidth=1, zorder=4
    )
    plot_country(
        ax,
        country=["AT", "LI", "DE", "IT", "FR"],
        edgecolor="none",
        facecolor="lightgrey",
        linewidth=1,
        zorder=3,
        alpha=1,
    )
    plot_lakes(ax, edgecolor="none", facecolor="lightblue", zorder=5)

    return fig, ax, cbax


def plot_standard_map(
    data_source=None,
    init_time=None,
    valid_time=None,
    prj=ccrs.AlbersEqualArea(8.222665776, 46.800663464),
    figsize=(10, 6.25),
):
    """standard map plot for sandbox
    defines figsize,  dpi, font, projection, map extent, colorbar axis
    ------
    data_source: string; name of data source, e.g., "COSMO-1E"; Optional
    init_time:   dt.datetime(); forecast initalization time; Optional
    valid_time:  [dt.datetime(),dt.datetime()]; start and end of valid time; Optional
    """
    # font settings
    fs = MAP_FONTSIZE
    params = {
        "font.size": fs,
        "legend.fontsize": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "mathtext.fontset": "dejavusans",
        "font.family": "sans-serif",
    }
    plt.rcParams.update(params)

    # projection
    # default prj = ccrs.AlbersEqualArea(8.222665776, 46.800663464)

    # create figure
    # fig, ax = plt.subplots(1, 1, figsize=(10,6), subplot_kw={'projection':prj}, dpi=150)
    fig = plt.figure(figsize=figsize, dpi=150)

    # add empty axis, to ensure constant plot sizes
    ax_empty = fig.add_axes((0, 0.1, 1, 0.8))  # left, bottom, witdth, height
    ax_empty.axis("off")

    # add main axis
    x0, y0, width, height = (
        0.01,
        0,
        0.88,
        1,
    )  # allow for enough space for widest legend / colorbar
    ax = fig.add_axes((x0, y0, width, height), projection=prj)

    # set extent (central point Switzerland 46.80111/8.22667 [Swiss Grid: 660158/183641])
    if prj is not None:
        ax.set_extent(MAP_EXTENT)

    # create axis for colorbar
    cbax = make_axes_locatable(ax).append_axes(
        "right", size="5%", pad=0.1, axes_class=plt.Axes
    )
    # cbax = fig.add_axes((x0+width*1.05,y0,0.05*width,height))

    # add data_source as left title
    if data_source:
        ax.set_title(data_source, loc="left", fontsize=fs)
    else:
        ax.set_title("PLACEHOLDER", loc="left", fontsize=fs, color="none")

    # add time info as string
    if valid_time:
        title_str = f'VALID {valid_time[0].strftime("%Y-%m-%dT%H")} $-$ {valid_time[1].strftime("%Y-%m-%dT%H UTC")}'
        if init_time:
            title_str = f'INIT {init_time.strftime("%Y-%m-%dT%H ")   } {title_str}'

        ax.set_title(title_str, loc="right", fontsize=fs)

    # return
    return fig, ax, cbax


# add cantons
def plot_canton(
    ax,
    canton="all",
    edgecolor="gray",
    facecolor="none",
    linewidth=0.5,
    zorder=0.5,
    linestyle="solid",
):
    """plot one or several cantons provided in the shapefile

    Parameters
    ----------
    canton: string, list of country abbreviations; available values: all, ZH, BE, ...
    """

    # read file
    # ch_shp_path =  "./ch_shapefile"
    reader = shpreader.Reader("%s/swissTLMRegio_KANTONSGEBIET_LV95.shp" % ch_shp_path)

    # all cantons
    if canton == "all":
        sel_cantons = [
            place
            for place in reader.records()
            if place.attributes["OBJEKTART"] == "Kanton"
            and place.attributes["ICC"] == "CH"
        ]
        for sel_canton in sel_cantons:
            shape_feature = cf.ShapelyFeature(
                [sel_canton.geometry],
                ccrs.epsg(2056),
                edgecolor=edgecolor,
                facecolor=facecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            ax.add_feature(shape_feature, zorder=zorder)

    # list of cantons
    elif type(canton) == list:
        sel_cantons = [
            place for place in reader.records() if place.attributes["NAME"] in canton
        ]
        for sel_canton in sel_cantons:
            shape_feature = cf.ShapelyFeature(
                [sel_canton.geometry],
                ccrs.epsg(2056),
                edgecolor=edgecolor,
                facecolor=facecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            ax.add_feature(shape_feature, zorder=zorder)

    # one canton
    elif type(canton) == str:
        sel_canton = [
            place for place in reader.records() if place.attributes["NAME"] == canton
        ][0]
        shape_feature = cf.ShapelyFeature(
            [sel_canton.geometry],
            ccrs.epsg(2056),
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.add_feature(shape_feature, zorder=zorder)


def plot_country(
    ax,
    country="all",
    edgecolor="white",
    facecolor="none",
    linewidth=1,
    zorder=0.5,
    **kwargs,
):
    """plot one or several countries provided in the shapefile

    Parameters
    ----------
    country: string, list of country iso code; available values: all, LI, AT, DE, CH, FR, IT
    """

    # ch_shp_path = "./ch_shapefile"
    reader3 = shpreader.Reader("%s/swissTLMRegio_LANDESGEBIET_LV95.shp" % ch_shp_path)

    # all countries
    if country == "all":
        for cntry in ["CH", "DE", "AT", "LI", "IT", "FR"]:
            sel_country = [
                place for place in reader3.records() if place.attributes["ICC"] == cntry
            ][0]
            shape_feature3 = cf.ShapelyFeature(
                [sel_country.geometry],
                ccrs.epsg(2056),
                edgecolor=edgecolor,
                facecolor=facecolor,
                linewidth=linewidth,
            )
            ax.add_feature(shape_feature3, zorder=zorder, **kwargs)

    # list of countries
    elif type(country) == list:
        for cntry in country:
            sel_country = [
                place for place in reader3.records() if place.attributes["ICC"] == cntry
            ][0]
            shape_feature3 = cf.ShapelyFeature(
                [sel_country.geometry],
                ccrs.epsg(2056),
                edgecolor=edgecolor,
                facecolor=facecolor,
                linewidth=linewidth,
            )
            ax.add_feature(shape_feature3, zorder=zorder, **kwargs)

    # one country only
    elif type(country) == str:
        sel_country = [
            place for place in reader3.records() if place.attributes["ICC"] == country
        ][0]
        shape_feature3 = cf.ShapelyFeature(
            [sel_country.geometry],
            ccrs.epsg(2056),
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
        )
        ax.add_feature(shape_feature3, zorder=zorder, **kwargs)


# plot lakes
def plot_lakes(ax, edgecolor="none", facecolor="lightblue", zorder=0.5, **kwargs):
    """plot Swiss lakes which have a certain size"""

    # read file
    # ch_shp_path = "./ch_shapefile"
    reader = shpreader.Reader("%s/Hydrography/swissTLMRegio_Lake.shp" % ch_shp_path)

    # get geometries
    geometry = reader.geometries()
    geometry = np.array([g for g in geometry])
    lakesize = np.array([a.area for a in reader.geometries()])
    geometry = geometry[lakesize > 1e7]  # default: 2e7
    shape_feature = cf.ShapelyFeature(
        geometry, ccrs.epsg(2056), edgecolor=edgecolor, facecolor=facecolor, **kwargs
    )

    # add to plot
    ax.add_feature(shape_feature, zorder=zorder)
