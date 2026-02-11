# -*- coding: utf-8 -*-
"""
Empirical calibration functions for spatially explicit vulnerability function calibration


"""

# other
import numpy as np
import pandas as pd
import sys, os  # argparse
from scipy import sparse, interpolate
from scipy.optimize import minimize
import datetime as dt
import copy
import json
import matplotlib
import matplotlib.pyplot as plt
import warnings
import xarray as xr


from climada.engine import ImpactCalc, Impact
from climada.entity import ImpactFuncSet, ImpfTropCyclone, ImpactFunc
from climada.engine.impact_data import emdat_impact_yearlysum, emdat_impact_event
from climada import CONFIG

sys.path.append(str(CONFIG.local_data.func_dir))

from utils import hazard_from_radar, get_emanuel_impf

###############################################################################
# Quantile match calibration
###############################################################################


def quantile_match_calib(
    haz,
    exp_in,
    imp_obs_rel,
    start_date,
    end_date,
    thresh_exp=0,
    metric="paa",
    date_mode="all_summers",
    n_members=1,
    bs_samples=None,
    method="sort",
    plot_level=1,
    save_path=None,
    bias_weight=0,
    verbose=True,
):
    """Calibrate an impact function by maching quantiles of hazard and
       RELATIVE impact (i.e. PAA or MDR)

    Args:
        haz (climada.Hazard or callable): hazard data or callable to get hazard file paths per date
        exp_in (climada.Exposure): exposure data
        imp_obs_rel (climada.engine.Impact): observed (relative) impact
            (i.e. PAA or MDR). Must have same spatial coords as exp_in
        start_date (str): start date in format 'YYYY-MM-DD'
        end_date (str): end date in format 'YYYY-MM-DD'
        thresh_exp (float, optional): Threshold for minimum exposure value to
            be considered. Defaults to 0.
        metric (str, optional): relative damage metric. Defaults to 'paa'.
        date_mode (str, optional): Date selection. Defaults to 'all_summers'.
        n_members (int, optional): number of ensemble members. Defaults to 1.
        bs_samples (int, optional): number of bootstrap samples. Defaults to None.
        method (str, optional): calculation method. Defaults to 'sort'.
        plot_level (int, optional): level of plotting. Defaults to 1.
        save_path (str, optional): path to save results. Defaults to None.
        bias_weight (int, optional): weight of bias in calibration (compared to MSE). Defaults to 0.
        verbose (bool, optional): print outputs. Defaults to True.

    Returns:
        tuple: tuple of impact function parameters, flexible impact function
            parameters, figure, impact
    """

    exp = copy.deepcopy(exp_in)

    if not method == "sort":
        raise NotImplementedError(
            f"method={method} not implemented. Use 'sort', which only works"
            f"with the same spatial and temporal extent in hazard and imp_obs."
            f"See subproj_A\impfct_hailcast\match_quantiles_hail_dmg_buildings_season_centroids.ipynb"
            f"for details on the 'quant' method"
        )

    # get a list of all dates
    if not date_mode == "all_summers":
        raise NotImplementedError(f"date_mode={date_mode} not implemented")

    date_list = pd.date_range(start_date, end_date, freq="D")
    date_list = date_list[(date_list.month >= 4) & (date_list.month <= 9)]
    n_days = len(date_list)

    # Select only impact between start and end date
    if dt.datetime.fromisoformat(start_date) > dt.datetime.fromordinal(
        imp_obs_rel.date[0]
    ) or dt.datetime.fromisoformat(end_date) < dt.datetime.fromordinal(
        imp_obs_rel.date[-1]
    ):
        print(
            "Not all impact dates are within the selected date range. Selecting subset of impact dates."
        )
        imp_obs_rel = imp_obs_rel.select(
            dates=(
                dt.datetime.fromisoformat(start_date).toordinal(),
                dt.datetime.fromisoformat(end_date).toordinal(),
            )
        )

    months = np.array([dt.datetime.fromordinal(d).month for d in imp_obs_rel.date])
    if any(months < 4) or any(months > 9):
        raise ValueError("Some impact dates are not within the summer season.")

    # assert consistent coordinates for exposure and impact
    np.testing.assert_array_almost_equal(
        # exp.gdf[["latitude", "longitude"]].values, imp_obs_rel.coord_exp
        np.array([exp.latitude, exp.longitude]).T,
        imp_obs_rel.coord_exp,
    )

    # select subset of centroids, which fullfill the min_exposure criterion
    if verbose:
        print(f"Centroids before subsetting: {exp.gdf.shape[0]}")
    condition = exp.gdf.value > thresh_exp
    exp.data = exp.data[condition]
    if verbose:
        print(f"Centroids after subsetting:   {exp.gdf.shape[0]}")

    # select a subset of centroids in impact
    imp_sel = imp_obs_rel.imp_mat[:, condition].toarray()
    if verbose:
        print(f"Impact after subsetting  (events, centroids):  {imp_sel.shape}")

    # get dates of events with reported damages
    date_events = [dt.date.fromordinal(date) for date in imp_obs_rel.date]
    n_events = len(date_events)
    if verbose:
        print(f"Number of events with reported impacts: {n_events}")

    # create imp_filled with dimensions (dates_in_period, selected_centroids) filled with zeros
    imp_filled = np.zeros((n_days, imp_sel.shape[1]))
    date_idx = [date_list.to_list().index(pd.Timestamp(date)) for date in date_events]
    # alternative with same results : [np.where(date_list==pd.Timestamp(date))[0][0] for date in date_events]
    imp_filled[date_idx, :] = imp_sel
    imp_sel = imp_filled  # overwrite imp_sel with imp_filled

    n_events = len(date_events)
    if verbose:
        print(f"\nImpact after filling dates with no damages: {imp_sel.shape}")
        print(f"{n_events} events in the period: {date_events[0]} - {date_events[-1]}")
        print(f"{n_days - n_events} of {n_days} days without damage")

    # get hazard intensity for all dates
    if callable(haz):
        hazard, dates_missing = get_hazard_from_files(
            haz, date_list, plot_level=plot_level
        )
        # Remove impact dates not in hazard data
        if len(dates_missing) > 0:
            print(
                f"\nRemove {len(dates_missing)} dates from imp_sel which are missing in hazard dataset."
            )
            idx_missing = [date_list.to_list().index(date) for date in dates_missing]
            print(f"Shape of imp_sel with all dates:             {imp_sel.shape}")
            imp_sel = np.delete(imp_sel, idx_missing, axis=0)
            print(f"Shape of imp_sel with missing dates removed: {imp_sel.shape}")

    else:
        hazard = haz

    # get corresponding hazard centroids
    exp.assign_centroids(hazard, threshold=100)
    haz_sel = hazard.intensity[:, exp.gdf.centr_HL].toarray()
    haz_sel_coord = hazard.centroids.coord[exp.gdf.centr_HL, :]

    # assert consistent coordinates for exposure and hazard (closest match)
    np.testing.assert_array_almost_equal(
        # exp.gdf[["latitude", "longitude"]].values, haz_sel_coord, decimal=1
        np.array([exp.latitude, exp.longitude]).T,
        haz_sel_coord,
        decimal=1,
    )

    if plot_level >= 2:
        # plot selected hazard data
        fig, ax = plt.subplots()
        c1 = ax.scatter(
            x=haz_sel_coord[:, 1],
            y=haz_sel_coord[:, 0],
            c=np.max(haz_sel, axis=0),
            vmin=0,
            vmax=45,
            s=10,
        )
        plt.colorbar(c1)
        ax.set(title="Selected hazard data (max value per centroid)")

    if n_members > 1:

        haz_sel_orig = copy.deepcopy(haz_sel)

        haz_sel = haz_sel.reshape((imp_sel.shape[0], n_members, imp_sel.shape[1]))
        # re-order dimensions
        haz_sel = np.moveaxis(haz_sel, 1, 2)
        # Version1:  haz_sel = haz_sel.reshape((imp_sel.shape[0],imp_sel.shape[1],n_members))

        # check if hazard and impact have the same number of timesteps and centroids
        assert haz_sel.shape[:2] == imp_sel.shape

        # DEBUG: Check that the data is re-shaped into the correct dimensions
        n_time = 0
        for i in range(len(hazard.event_id)):
            if np.mod(i, n_members) == 0 and i > 0:
                n_time += 1
            ens_member = i - n_members * n_time

            # print(f"time: {n_time},ens member: {ens_member}")
            # print(hazard.event_name[i])
            # if not haz_test[n_time,:,ens_member].max() == 0:
            #     print(haz_sel_orig[i,:].max())

            np.testing.assert_array_equal(
                haz_sel_orig[i, :], haz_sel[n_time, :, ens_member]
            )

    else:
        # check if hazard and impact have the same number of timesteps and centroids
        assert haz_sel.shape == imp_sel.shape

    # flatten and sort the two arrays
    haz_vals_sorted, imp_vals_sorted = get_sorted_data(haz_sel, imp_sel, n_members)

    # get impact functions from quantile matching
    params, params_flex, impf_em, impf_em_flex, df_impf, x_max_obs = get_impfs(
        haz_vals_sorted,
        imp_vals_sorted,
        metric,
        min_v_thresh="min_haz",
        verbose=verbose,
        bias_weight=bias_weight,
    )

    # get impact function with flexible lower threshold
    _, params_flex_thresh, _, impf_em_flex_thresh, _, _ = get_impfs(
        haz_vals_sorted,
        imp_vals_sorted,
        metric,
        min_v_thresh="none",
        verbose=verbose,
        bias_weight=bias_weight,
    )

    if save_path:
        for fp, p in zip(
            [
                f"{save_path}.json",
                f"{save_path}_flex.json",
                f"{save_path}_flex_thresh.json",
            ],
            [params, params_flex, params_flex_thresh],
        ):
            save_params(fp, p)

    # Plot the resulting impact functions
    if plot_level >= 1:
        fig = plot_quant_match_impf(
            haz_vals_sorted,
            imp_vals_sorted,
            impf_em,
            impf_em_flex,
            df_impf,
            params_flex,
            thresh_exp,
            date_mode,
            start_date,
            end_date,
            x_max_obs,
            metric,
        )
        ax = fig.get_axes()[0]
    else:
        fig = None

    # get bootstrapped hazard and impact values
    if bs_samples is not None:

        bootstrapped_haz_sorted, bootstrapped_imp_sorted, haz_counts = (
            get_bootstrapped_haz_imp_vals(
                haz_sel,
                imp_sel,
                bs_samples,
                n_members,
                return_full_array=False,
                haz_decimal=1,
            )
        )
        if save_path:
            np.savez(
                f"{save_path}_bs.npz",
                bs_haz=bootstrapped_haz_sorted,
                bs_imp=bootstrapped_imp_sorted,
            )

        # get quantile of bootstrap sample
        q_arrs = []
        for quantile, scale_bounds in zip(
            [0.05, 0.95], [[0, params_flex["scale"]], [params_flex["scale"], 1]]
        ):
            # Note: scale_bound are ignored for now

            q_arr = np.nanquantile(
                numpy_ffill(bootstrapped_imp_sorted), quantile, axis=0
            )
            (
                paramsBS,
                params_flexBS,
                impf_emBS,
                impf_em_flexBS,
                df_impfBS,
                x_max_obsBS,
            ) = get_impfs(
                bootstrapped_haz_sorted[0, :],
                q_arr,
                metric,
                min_v_thresh="min_haz",
                value_counts=haz_counts,
                verbose=False,
                scale_bounds=None,
                bias_weight=bias_weight,
            )
            print(f"Q {quantile*100:.0f}%: {params_flexBS}")
            # ax.plot(bootstrapped_haz_sorted[0,:] ,q_arr,color='black',alpha=0.9)
            if plot_level >= 1:
                ax.plot(
                    impf_em_flexBS.intensity,
                    impf_em_flexBS.mdd * 100,
                    color="tab:blue",
                    linestyle="dashed",
                    label=f"{int(100*quantile)}%-quantile impact function",  # label=f"Emanuel-type Q{quantile}  power={params_flexBS['power']:.1f} (opt)",
                )
            q_arrs.append(q_arr)
            if save_path:
                save_params(f"{save_path}_Q{quantile:.2f}_flex.json", params_flexBS)

        if plot_level >= 1:
            ax.fill_between(
                bootstrapped_haz_sorted[0, :],
                q_arrs[0] * 100,
                q_arrs[1] * 100,
                color="grey",
                alpha=0.2,
                label="5-95% quantile",
            )

            for i in range(bs_samples):
                not_nan = ~np.isnan(bootstrapped_imp_sorted[i, :])
                ax.plot(
                    bootstrapped_haz_sorted[i, :][not_nan],
                    bootstrapped_imp_sorted[i, :][not_nan] * 100,
                    linewidth=1,
                    linestyle="-",
                    color="grey",
                    alpha=0.1,
                )

            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

            # Debug: Check the spatial distribution of bootstrapped samples to make sure they are correct
            if plot_level >= 4:
                plot_spatial_distribution_of_bs_samples(
                    haz_sel, imp_sel, bs_samples, n_members, haz_sel_coord, ax
                )
            
        fig.savefig(f"{save_path}_plot.png", dpi=300, bbox_inches="tight")


        return (
            params,
            params_flex,
            fig,
            haz_vals_sorted,
            bootstrapped_haz_sorted,
            bootstrapped_imp_sorted,
            haz_counts,
        )

    return params, params_flex, fig, haz_vals_sorted


def plot_spatial_distribution_of_bs_samples(
    haz_sel, imp_sel, bs_samples, n_members, haz_sel_coord, ax
):
    """Function for debugging only. Plot the spatial distribution of
    bootstrapped samples to make sure they are correct.
    Only works for forecast data with n_members > 1"""
    bootstrapped_haz, bootstrapped_imp = get_bootstrapped_haz_imp_vals(
        haz_sel, imp_sel, bs_samples, n_members, return_full_array=True
    )
    for i in range(bs_samples):
        haz_vals_bs, imp_vals_bs = get_sorted_data(
            bootstrapped_haz[i], bootstrapped_imp[i], n_members
        )
        ax.plot(
            haz_vals_bs,
            imp_vals_bs * 100,
            linewidth=1,
            linestyle="-",
            color="salmon",
            alpha=0.1,
        )

    i, j = np.where(
        bootstrapped_haz[0, :].sum(axis=1)
        > np.quantile(bootstrapped_haz[0, :].sum(axis=1), 0.9)
    )
    for count in range(min(20, len(i))):
        i_now = i[count]
        j_now = j[count]
        haz_now = bootstrapped_haz[0, i_now, :, j_now]
        # plot selected hazard data
        # c1 = plt.tricontourf(haz_sel_coord[:,1], haz_sel_coord[:,0], np.max(haz_sel,axis=0), levels=50)
        fig, ax = plt.subplots()
        c1 = ax.scatter(
            x=haz_sel_coord[:, 1],
            y=haz_sel_coord[:, 0],
            c=haz_now,
            vmin=0,
            vmax=45,
            s=5,
        )
        plt.colorbar(c1)
        ax.set(title="Selected hazard data (max value per centroid)")


############# Helper functions for quantile match calibration ################
def save_params(path, params):
    with open(path, "w") as file:
        json.dump(params, file)


def get_sorted_data(
    haz_sel,
    imp_sel,
    n_members=1,
    subsampling=False,
    subsample_decimals=1,
    haz_vals_subsampled=None,
):
    """Get sorted hazard and impact data. If n_members > 1, subsample
    hazard data to quantile match centroids correctly

    Args:
        haz_sel (np.array): selected hazard values
        imp_sel (np.array): selected impact values
        n_members (int, optional): number of ensemble members. Defaults to 1.
        subsampling (bool, optional): subsample data to hazard values with given
          precision (reduces memory usage). Defaults to False.
        subsample_decimals (int, optional): number of decimals for subsampling.
            Defaults to 1.
        haz_vals_subsampled (np.array, optional): subsampled hazard values

    Returns:
        tuple: tuple of sorted hazard and impact data
    """

    # flatten and sort the two arrays
    imp_vals_sorted = np.sort(imp_sel.flatten())
    haz_vals_sorted = np.sort(haz_sel.flatten())

    if n_members > 1:
        # subsample hazard data to quantile match centroids correctly
        # select every n-th value of haz_vals_sorted
        # start at n_members/2 to get the middle value of each "n_memeber group"
        haz_vals_sorted = haz_vals_sorted[int(n_members / 2) :: n_members]

    if subsampling:
        haz_vals_sorted = haz_vals_sorted.round(decimals=subsample_decimals)
        if haz_vals_subsampled is None:
            # Subsample data so that only one value per 0.1 of hazard intensity is used
            haz_vals_subsampled = np.unique(haz_vals_sorted)
        else:
            # assert that the precision of the given subsampled values is correct, and that they are sorted
            assert np.all(
                np.round(np.sort(haz_vals_subsampled), decimals=subsample_decimals)
                == haz_vals_subsampled
            )

        imp_vals_subsampled = np.array(
            [
                np.mean(imp_vals_sorted[haz_vals_sorted == haz])
                for haz in haz_vals_subsampled
            ]
        )

        assert len(haz_vals_subsampled) == len(imp_vals_subsampled)
        return haz_vals_subsampled, imp_vals_subsampled

    else:
        assert len(haz_vals_sorted) == len(imp_vals_sorted)
        return haz_vals_sorted, imp_vals_sorted


def get_bootstrapped_haz_imp_vals(
    haz_sel, imp_sel, n_samples, n_members=1, return_full_array=False, haz_decimal=1
):
    """Get bootstrapped hazard and impact values

    Args:
        haz_sel (np.array): selected hazard values
        imp_sel (np.array): selected impact values
        n_samples (int): number of bootstrap samples
        return_full_array (bool, optional): return full array of bootstrapped
            values (needs more memory; for debugging mainly). Defaults to False.
        haz_decimal (int, optional): number of decimals for unique hazard values. Defaults to 1.

    Returns:
        tuple: tuple of bootstrapped hazard and impact values
    """

    # Set random seed for reproducibility
    rg = np.random.default_rng(seed=123)

    if return_full_array:
        # get bootstrapped hazard and impact values (without ensemble memebrs)
        if n_members == 1:  # and haz_sel.ndim == 2 and imp_sel.ndim == 2:
            bootstrapped_haz = np.zeros((n_samples, haz_sel.shape[0], haz_sel.shape[1]))
            bootstrapped_imp = np.zeros((n_samples, imp_sel.shape[0], imp_sel.shape[1]))

            for i in range(n_samples):
                if i == 0:
                    # select original data
                    sel_sample_dates = np.arange(haz_sel.shape[0])
                else:
                    sel_sample_dates = rg.choice(
                        haz_sel.shape[0], size=haz_sel.shape[0], replace=True
                    )
                bootstrapped_haz[i, :, :] = haz_sel[sel_sample_dates, :]

                bootstrapped_imp[i, :, :] = imp_sel[sel_sample_dates, :]

        # get bootstrapped hazard and impact values (with ensemble members)
        elif haz_sel.ndim == 3 and imp_sel.ndim == 2:
            # hazard dims: (n_events, n_centroids, n_members)
            # impact dims: (n_events, n_centroids)
            bootstrapped_haz = np.zeros(
                (n_samples, haz_sel.shape[0], haz_sel.shape[1], haz_sel.shape[2])
            )
            bootstrapped_imp = np.zeros((n_samples, imp_sel.shape[0], imp_sel.shape[1]))

            for i in range(n_samples):
                if i == 0:
                    # select original data
                    sel_sample_dates = np.arange(haz_sel.shape[0])
                else:
                    sel_sample_dates = rg.choice(
                        haz_sel.shape[0], size=haz_sel.shape[0], replace=True
                    )
                bootstrapped_haz[i, :, :, :] = haz_sel[sel_sample_dates, :, :]

                bootstrapped_imp[i, :, :] = imp_sel[sel_sample_dates, :]

        return bootstrapped_haz, bootstrapped_imp

    else:
        # unique hazard values with a precision of 0.1 (1 decimal)
        haz_vals = np.round(haz_sel, decimals=haz_decimal)
        unique_haz_vals, haz_counts = np.unique(haz_vals, return_counts=True)

        # count occurences of each unique hazard value

        # get bootstrapped hazard and impact values (without ensemble memebrs)
        bootstrapped_haz = np.zeros((n_samples, len(unique_haz_vals)))
        bootstrapped_imp = np.zeros((n_samples, len(unique_haz_vals)))

        for i in range(n_samples):
            if i == 0:
                sel_sample_dates = np.arange(haz_sel.shape[0])
            else:
                sel_sample_dates = rg.choice(
                    haz_sel.shape[0], size=haz_sel.shape[0], replace=True
                )
            haz_vals_sorted, imp_vals_sorted = get_sorted_data(
                haz_sel[sel_sample_dates, :],
                imp_sel[sel_sample_dates, :],
                n_members=n_members,
                subsampling=True,
                subsample_decimals=haz_decimal,
                haz_vals_subsampled=unique_haz_vals,
            )
            bootstrapped_haz[i, :] = haz_vals_sorted
            bootstrapped_imp[i, :] = imp_vals_sorted

        return bootstrapped_haz, bootstrapped_imp, haz_counts


def get_impfs(
    haz_vals_sorted,
    imp_vals_sorted,
    metric,
    min_v_thresh="none",
    value_counts=None,
    scale_bounds=None,
    verbose=False,
    bias_weight=0,
):
    """Get impact functions from quantile matching

    Args:
        haz_vals_sorted (np.array): sorted hazard values
        imp_vals_sorted (np.array): sorted RELATIVE impact values (PAA or MDR)
        metric (str): relative damage metric
        min_v_thresh (str, optional): minimum value for v_thresh. Defaults to 'none'.
            if 'min_haz' use minimum hazard value with nonzero impact
        value_counts (np.array, optional): counts of unique hazard values. Defaults to None.
        scale_bounds (tuple, optional): bounds for scale parameter. Defaults to None.
        verbose (bool, optional): print verbose output. Defaults to False.
        bias_weight (float, optional): weight for bias in comparision to MSE,
                                        when optimizing the impact function. Defaults to 0.

    Returns:
        tuple: tuple of impact function parameters, flexible impact function
            parameters, figure, hazard values
    """

    # create dataframe for fit_emanuel_to_impf function
    both_zero = (imp_vals_sorted == 0) & (haz_vals_sorted == 0)
    df_impf = pd.DataFrame(
        {"intensity": haz_vals_sorted, metric.upper(): imp_vals_sorted}
    ).set_index("intensity")

    if value_counts is None:
        df_impf["count_cell"] = 1  # give equal weight (=1) to each row
    else:
        df_impf["count_cell"] = value_counts

    opt_var = metric.upper()  # PAA or MDR
    df_impf = df_impf[~both_zero]  # remove double zeros for faster calculation

    # edit dataframe to have one value per FULL INTEGER value of hazard intensity
    column_mapping = {col: "mean" for col in df_impf.columns}
    column_mapping["count_cell"] = "sum"
    df_impf["haz_rounded"] = df_impf.index.values.round().astype(int)
    df_1steps = df_impf.groupby("haz_rounded").agg(column_mapping)

    # Define parameter bounds for Emanuel-type impact function
    x_max_obs = int(df_1steps.index.max())
    if min_v_thresh == "min_haz":
        v_thresh = haz_vals_sorted[imp_vals_sorted > 0].min()
    elif min_v_thresh == "none":
        v_thresh = 0
    else:
        raise ValueError(f"Invalid value for min_v_thres {min_v_thresh}")

    y_bounds = [
        v_thresh * 1.01,
        x_max_obs,
    ]  # add 1% to avoid v_tresh==v_half in starting array
    v_tresh_bounds = [v_thresh, x_max_obs]

    if scale_bounds is None:
        scale_bounds = [imp_vals_sorted.max() / 10, min(1, imp_vals_sorted.max() * 10)]

    pbounds = {
        "v_thresh": v_tresh_bounds,
        "v_half": y_bounds,
        "scale": scale_bounds,
        "power": (3, 3),
    }

    param_start = {
        "v_thresh": v_thresh,
        "v_half": (x_max_obs - v_thresh) / 2 + v_thresh,
        "scale": min(1, imp_vals_sorted.max()),
        "power": 3,
    }

    # fit Emanuel-type sigmoid function to impact function
    params, _, _ = fit_emanuel_impf_to_emp_data(
        df_1steps,  # alternative: df_impf
        pbounds,
        opt_var,
        plot=False,
        max_iter=1000,
        verbose=verbose,
        param_start=list(param_start.values()),
        bias_weight=bias_weight,
    )

    impf_em = get_emanuel_impf(
        **params, impf_id=1, intensity=np.arange(0, x_max_obs, 1)
    )

    pbounds["power"] = (1, 20)  # flexible power law in Emanuel-type function
    param_start["power"] = 5
    params_flex, _, _ = fit_emanuel_impf_to_emp_data(
        df_1steps,  # alternative: df_impf
        pbounds,
        opt_var,
        plot=False,
        max_iter=1000,
        verbose=verbose,
        param_start=list(param_start.values()),
        bias_weight=bias_weight,
    )

    impf_em_flex = get_emanuel_impf(
        **params_flex, impf_id=1, intensity=np.arange(0, x_max_obs, 1)
    )

    return params, params_flex, impf_em, impf_em_flex, df_impf, x_max_obs


def get_hazard_from_files(haz, date_list, plot_level=1):
    """Get hazard intensity for all dates in date_list

    Args:
        haz (callable): callable to get hazard file paths per date
        date_list (pandas.DatetimeIndex): list of dates
        plot_level (int, optional): level of plotting. Defaults to 1.

    Returns:
        climada.hazard: hazard object
    """
    f_haz = []
    dates_missing = []
    for date in date_list:
        file_path = haz(date)
        # check if file exists
        if os.path.exists(file_path):
            # if the file exists, write the path to the f_haz list
            f_haz.append(file_path)
        else:
            # if the file does not exist, write the date to the f_haz_missing list
            dates_missing.append(date)
    print(f"Hazard files available: {len(f_haz)}")
    print(f"Hazard files missing:   {len(dates_missing)}")

    # load data
    hazard = hazard_from_radar(
        f_haz,
        varname="DHAIL_MX",
        time_dim="time",
        ensemble_dim="epsd_1",
        spatial_dims=["x_1", "x_2"],
    )
    print(
        f"Hazard dimensions (n_time*n_members, n_centroids): {hazard.intensity.shape}"
    )
    if plot_level >= 2:
        hazard.plot_intensity(0, vmin=0, vmax=45)

    return hazard, dates_missing


############# Plotting functions for quantile match calibration ###############
def plot_quant_match_impf(
    haz_vals_sorted,
    imp_vals_sorted,
    impf_em,
    impf_em_flex,
    df_impf,
    p2,
    thresh_exp,
    date_mode,
    start_date,
    end_date,
    x_max_obs,
    metric="paa",
):
    """Plot impact function from quantile matching

    Args:
        haz_vals_sorted (np.array): sorted hazard values
        imp_vals_sorted (np.array): sorted RELATIVE impact values (PAA or MDR)
        impf_em (climada.entity.Impf): Optimized impact function with fixed power=3
        impf_em_flex (climada.entity.Impf): Optimized impact function with flexible power (1 to 20)
        df_impf (pd.DataFrame): calibration data
        p2 (dict): parameters of flexible Emanuel-type impact function
        thresh_exp (float): Min. considered exposure value
        date_mode (str): Date selection mode. Defaults to 'all_summers'
        start_date (str): start date
        end_date (str): End date
        x_max_obs (float): Maximum observed relative impact (PAA or MDR)
        metric (str, optional): relative damage metric. Defaults to 'paa'.

    Returns:
        plt.Figure: Plot of impact function
    """

    if metric == "paa":
        ylabel = "Percent of Assets Affected [%]"
    elif metric == "mdr":
        ylabel = "Mean Damage Ratio [%]"
    else:
        raise ValueError(f"Invalid metric {metric}")

    fig, ax = plt.subplots()

    ax.plot(
        haz_vals_sorted,
        imp_vals_sorted * 100,
        linewidth=2,
        linestyle="-",
        color="k",
        label="quantile matching",  # label="member mean",
    )
    # ax.plot(
    #     impf_em.intensity, impf_em.mdd*100, color="lightgreen", label="Emanuel-type (opt)"
    # )
    ax.plot(
        impf_em_flex.intensity,
        impf_em_flex.mdd * 100,
        color="tab:blue",
        label=f"impact function",  # label=f"Emanuel-type  power={p2['power']:.1f} (opt)",
    )
    # text_size = f"No. of non-zero cells = {int(imp_vals_sorted.shape[0])}"

    ax.set(
        xlim=(0, x_max_obs * 1.05),
        xlabel="HAILCAST DHAIL$_\mathrm{max}$ [mm]",
        ylabel=ylabel,
    )
    text = (
        f"centroids with exposure > {thresh_exp}\n{date_mode} "
        f"from {start_date} to {end_date}\n\n"
    )

    # ax.text(0.02, 0.98, text + text_size, va="top", ha="left", transform=ax.transAxes)

    # add legend below the figure
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    ax2 = ax.twinx()
    color_number_cells = "firebrick"
    _ = ax2.hist(df_impf.index, bins=40, alpha=0.3, color=color_number_cells, log=True)
    ax2.tick_params(axis="y", which="major", colors=color_number_cells)
    ax2.tick_params(axis="y", which="minor", colors=color_number_cells)
    ax2.yaxis.label.set_color(color_number_cells)
    ax2.set(ylabel="Number of cells")
    return fig


def numpy_ffill(arr):
    """Forward fill NaNs in 2D array along axis 1."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def fit_emanuel_impf_to_emp_data(
    emp_df,
    pbounds,
    opt_var="MDR",
    options=None,
    optimizer="Nelder-Mead",
    plot=True,
    param_start=None,
    verbose=True,
    max_iter=500,
    bias_weight=0,
):
    """Fit emanuel-type impact function to empirical data

    Args:
        emp_df (pd.DataFrame): DF with empirical data, including 'MDR', 'PAA', and 'count_cell'
        pbounds (dict): dictionary of parameter bounds
        opt_var (str, optional): Variable from emp_df to fit data to. Defaults to 'MDR'.
        options (dict, optional): Additional options for the Bayesian optimizer. Defaults to None.
        optimizer (str, optional): Choice of optimizer. Defaults to 'Nelder-Mead'.
        plot (bool, optional): Whether or not to plot the data. Defaults to True.
        param_start (dict, optional): Initial parameter values. Defaults to None.
        verbose (bool, optional): Whether or not to print the results. Defaults to True.
        max_iter (int, optional): Maximum number of iterations for the optimizer. Defaults to 500.
        bias_weight (int, optional): Weight of the bias in the optimization. Defaults to 0.
                                    if >0: weight MSE with |impact bias [%]|. i.e. if the impact bias is 5%
                                    and the weight is 1, the MSE is multiplied by 1.05

    Raises:
        ValueError: if get_emanuel_impf returns an error except for v_half <= v_thresh

    Returns:
        tuple: Parameters, optimizer object, and impact function
    """
    if not emp_df.index.name:
        raise ValueError("Careful, emp_df.index has no name. Double check if the \
                            index corresponds to the intensity unit!")

    def weighted_MSE(**param_dict):
        try:
            impf = get_emanuel_impf(**param_dict, intensity=emp_df.index.values)
        except ValueError as e:
            if "v_half <= v_thresh" in str(e):
                # if invalid input to Emanuel_impf (v_half <= v_thresh), return zero
                return np.inf
            else:
                raise ValueError(f"Unknown Error in init_impf:{e}. Check inputs!")

        imp_mod = impf.mdd * impf.paa
        SE = np.square(imp_mod - emp_df[opt_var])

        SE_weigh = SE * emp_df.count_cell
        SE_weigh_noZero = SE_weigh[emp_df.index.values != 0]
        MSE = np.mean(
            SE_weigh_noZero
        )  # mean over all intensity values (equivalent result with sum)

        if bias_weight > 0:
            # #select only values with nonzero hazard and not zero impact for both empirical and model
            # NOTE: less intuitive than equivalent impact
            # nonzero = (emp_df.index.values!=0) & ~((emp_df[opt_var]==0) & (imp_mod==0) )
            # bias = imp_mod - emp_df[opt_var]
            # weighted_bias = (bias[nonzero]*emp_df.count_cell[nonzero]).sum()/emp_df.count_cell[nonzero].sum()
            # print(f'weighted bias: {weighted_bias:.2e}')

            # calculate equivalent impacts
            imp_mod_sum = (imp_mod * emp_df.count_cell).sum()
            imp_emp_sum = (emp_df[opt_var] * emp_df.count_cell).sum()
            imp_bias = (imp_mod_sum - imp_emp_sum) / imp_emp_sum
            # print(f'impact difference: {imp_bias*100:.2e}%')

            # add bias to MSE
            MSE = MSE * (1 + abs(imp_bias) * bias_weight)

        return MSE

    if optimizer == "Nelder-Mead" or "trust-constr":
        if param_start is None:
            # use mean of bounds as starting point
            param_means = [(v[0] + v[1]) / 2 for v in pbounds.values()]
        elif type(param_start) == dict:
            param_means = param_start.values()
        else:
            param_means = param_start
        param_dict = dict(zip(pbounds.keys(), param_means))
        bounds = [(v[0], v[1]) for v in pbounds.values()]
        x0 = list(param_dict.values())

        # define function that returns the MSE, with an array x as input
        def mse_x(x):
            param_dict_temp = dict(zip(param_dict.keys(), x))
            # return -weighted_inverse_MSE(**param_dict_temp)
            return weighted_MSE(**param_dict_temp)

        if optimizer == "trust-constr":
            print(param_dict, bounds)
            np.testing.assert_array_equal(
                (list(pbounds.keys())[:2]), ["v_thresh", "v_half"]
            )
            cons = (
                {
                    "type": "ineq",
                    "fun": lambda x: x[1] - x[0],
                },  # v_half-v_tresh is to be non-negative
                {"type": "ineq", "fun": lambda x: x[2]},
            )  # scale is to be non-negative
            options = None

        elif optimizer == "Nelder-Mead":
            cons = None
            options = {"disp": verbose, "maxiter": max_iter}

        elif optimizer == "DIRECT":  # seems to also work
            from scipy.optimize import direct

            raise NotImplementedError("DIRECT optimizer not yet implemented")

        res = minimize(
            mse_x,
            x0,
            bounds=bounds,
            constraints=cons,
            # jac=gradient_respecting_bounds(bounds, mse_x) if optimizer == 'trust-constr' else None,
            # method='SLSQP',
            method=optimizer,
            options=options,
        )

        optimizer = res
        param_dict_result = dict(zip(param_dict.keys(), res.x))
        impf = init_impf("emanuel_HL", param_dict_result, emp_df.index.values)[0]

        if verbose:
            print(param_dict_result)
            # if bias_weight>0:
            imp_mod = impf.mdd * impf.paa
            imp_mod_sum = (imp_mod * emp_df.count_cell).sum()
            imp_emp_sum = (emp_df[opt_var] * emp_df.count_cell).sum()
            imp_bias = (imp_mod_sum - imp_emp_sum) / imp_emp_sum
            print(f"Equiv. impact bias: {imp_bias*100:.3f}%")

        if plot:
            ax = impf.plot(zorder=3)
            title = [
                (
                    f"{key}: {param_dict_result[key]:.2f}"
                    if param_dict_result[key] > 0.1
                    else f"{key}: {param_dict_result[key]:.2e}"
                )
                for key in param_dict_result.keys()
            ]
            ax.set(ylim=(0, max(impf.mdd * 100)), title=title)
            # add empirical function to plot
            ax.plot(emp_df.index, emp_df[opt_var] * 100, label=f"Empirical {opt_var}")
            plt.legend()

    return param_dict_result, optimizer, impf


def init_impf(
    impf_name_or_instance, param_dict, intensity_range, df_out=pd.DataFrame(index=[0])
):
    """create an ImpactFunc based on the parameters in param_dict using the
    method specified in impf_parameterisation_name and document it in df_out.

    Parameters
    ----------
    impf_name_or_instance : str or ImpactFunc
        method of impact function parameterisation e.g. 'emanuel' or an
        instance of ImpactFunc
    param_dict : dict, optional
        dict of parameter_names and values
        e.g. {'v_thresh': 25.7, 'v_half': 70, 'scale': 1}
        or {'mdd_shift': 1.05, 'mdd_scale': 0.8, 'paa_shift': 1, paa_scale': 1}
    intensity_range : array
        tuple of 3 intensity numbers along np.arange(min, max, step)
    Returns
    -------
    imp_fun : ImpactFunc
        The Impact function based on the parameterisation
    df_out : DataFrame
        Output DataFrame with headers of columns defined and with first row
        (index=0) defined with values. The impact function parameters from
        param_dict are represented here.
    """
    impact_func_final = None
    if isinstance(impf_name_or_instance, str):
        if impf_name_or_instance == "emanuel":
            impact_func_final = ImpfTropCyclone.from_emanuel_usa(**param_dict)
            impact_func_final.haz_type = "TC"
            impact_func_final.id = 1
            df_out["impact_function"] = impf_name_or_instance
        if impf_name_or_instance == "emanuel_HL":
            impact_func_final = get_emanuel_impf(
                **param_dict, intensity=intensity_range, haz_type="HL"
            )
            df_out["impact_function"] = impf_name_or_instance
        elif impf_name_or_instance == "sigmoid_HL":
            assert (
                "L" in param_dict.keys()
                and "k" in param_dict.keys()
                and "x0" in param_dict.keys()
            )
            impact_func_final = ImpactFunc.from_sigmoid_impf(
                **param_dict, intensity=intensity_range
            )  # ,haz_type='HL')
            impact_func_final.haz_type = "HL"
            if intensity_range[0] == 0 and not impact_func_final.mdd[0] == 0:
                warnings.warn(
                    "sigmoid impact function has non-zero impact at intensity 0. Setting impact to 0."
                )
                impact_func_final.mdd[0] = 0
            df_out["impact_function"] = impf_name_or_instance

    elif isinstance(impf_name_or_instance, ImpactFunc):
        impact_func_final = change_impf(impf_name_or_instance, param_dict)
        df_out["impact_function"] = (
            "given_" + impact_func_final.haz_type + str(impact_func_final.id)
        )
    for key, val in param_dict.items():
        df_out[key] = val
    return impact_func_final, df_out


def change_impf(impf_instance, param_dict):
    """apply a shifting or a scaling defined in param_dict to the impact
    function in impf_istance and return it as a new ImpactFunc object.

    Parameters
    ----------
    impf_instance : ImpactFunc
        an instance of ImpactFunc
    param_dict : dict
        dict of parameter_names and values (interpreted as
        factors, 1 = neutral)
        e.g. {'mdd_shift': 1.05, 'mdd_scale': 0.8,
        'paa_shift': 1, paa_scale': 1}

    Returns
    -------
    ImpactFunc : The Impact function based on the parameterisation
    """
    ImpactFunc_new = copy.deepcopy(impf_instance)
    # create higher resolution impact functions (intensity, mdd ,paa)
    paa_func = interpolate.interp1d(
        ImpactFunc_new.intensity, ImpactFunc_new.paa, fill_value="extrapolate"
    )
    mdd_func = interpolate.interp1d(
        ImpactFunc_new.intensity, ImpactFunc_new.mdd, fill_value="extrapolate"
    )
    temp_dict = dict()
    temp_dict["paa_intensity_ext"] = np.linspace(
        ImpactFunc_new.intensity.min(),
        ImpactFunc_new.intensity.max(),
        (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1,
    )
    temp_dict["mdd_intensity_ext"] = np.linspace(
        ImpactFunc_new.intensity.min(),
        ImpactFunc_new.intensity.max(),
        (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1,
    )
    temp_dict["paa_ext"] = paa_func(temp_dict["paa_intensity_ext"])
    temp_dict["mdd_ext"] = mdd_func(temp_dict["mdd_intensity_ext"])
    # apply changes given in param_dict
    for key, val in param_dict.items():
        field_key, action = key.split("_")
        if action == "shift":
            shift_absolut = ImpactFunc_new.intensity[
                np.nonzero(getattr(ImpactFunc_new, field_key))[0][0]
            ] * (val - 1)
            temp_dict[field_key + "_intensity_ext"] = (
                temp_dict[field_key + "_intensity_ext"] + shift_absolut
            )
        elif action == "scale":
            temp_dict[field_key + "_ext"] = np.clip(
                temp_dict[field_key + "_ext"] * val, a_min=0, a_max=1
            )
        else:
            raise AttributeError(
                "keys in param_dict not recognized. Use only:"
                "paa_shift, paa_scale, mdd_shift, mdd_scale"
            )

    # map changed, high resolution impact functions back to initial resolution
    ImpactFunc_new.intensity = np.linspace(
        ImpactFunc_new.intensity.min(),
        ImpactFunc_new.intensity.max(),
        (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1,
    )
    paa_func_new = interpolate.interp1d(
        temp_dict["paa_intensity_ext"], temp_dict["paa_ext"], fill_value="extrapolate"
    )
    mdd_func_new = interpolate.interp1d(
        temp_dict["mdd_intensity_ext"], temp_dict["mdd_ext"], fill_value="extrapolate"
    )
    ImpactFunc_new.paa = paa_func_new(ImpactFunc_new.intensity)
    ImpactFunc_new.mdd = mdd_func_new(ImpactFunc_new.intensity)
    return ImpactFunc_new
