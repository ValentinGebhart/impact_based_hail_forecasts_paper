# from plot_util_funcs import plot_standard_map, plot_canton, plot_country, plot_lakes
# from climada import CONFIG
from climada.engine import Impact
from climada.entity import Exposures, ImpactFunc  # , ImpactFuncSet
from climada.hazard import Hazard

# import sys
# sys.path.append(str(CONFIG.local_data.func_dir))
# sys.path.append('/Users/vgebhart/gitprojects/scClim/subproj_D/sandbox2')
# import scClim.hail_climada as fct
# import scClim as sc
# # import sandbox_plot_functions as scw
import numpy as np
import pandas as pd
import datetime as dt

# import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

# from itertools import product
# from scipy.optimize import minimize
# import cartopy.io.shapereader as shpreader
import geopandas as gpd
from scipy import sparse
import warnings


#############################
### util functions for io ###
#############################


# load exposure
def load_exposure(f_damage, variable, thresh_exp):

    exp = read_xr_exposure(f_damage, variable, "count")
    # exp.gdf["impf_"] = np.full_like(exp.gdf.length.size, 1)

    # select subset of centroids
    print(f"Centroids before subsetting: {exp.gdf.shape[0]}")
    condition = exp.gdf.value > thresh_exp
    exp.set_gdf(exp.gdf[condition])
    print(f"Centroids after subsetting:   {exp.gdf.shape[0]}")
    return exp, condition


# read exposure
def read_xr_exposure(nc_file, var_name, val_unit="CHF"):
    """Read exposure from netCDF file"""
    if isinstance(nc_file, str):
        nc = xr.open_dataset(nc_file)[var_name]
    elif isinstance(nc_file, xr.Dataset):
        nc = nc_file[var_name]
    nc = nc.rename("value")
    df_exp = nc.to_dataframe()
    if "lon" in df_exp.columns:
        df_exp = df_exp.rename(columns={"lon": "longitude", "lat": "latitude"})
    gdf_exp = gpd.GeoDataFrame(
        df_exp,
        geometry=gpd.points_from_xy(df_exp.longitude, df_exp.latitude, crs="EPSG:4326"),
    )
    exp = Exposures(gdf_exp, value_unit=val_unit)
    # add 'impf_' columns if it was not initialized already
    if not any(column.startswith("impf_") for column in exp.gdf.columns):
        exp.gdf["impf_"] = np.full_like(exp.gdf.shape[0], 1)
    exp.check()

    return exp


def read_xr_impact(
    nc_file,
    var_name,
    spatial_dims=["chy", "chx"],
    time_dim="date",
    unit="CHF",
    years=None,
):
    """Read impact from netCDF file"""
    if isinstance(nc_file, str):
        nc = xr.open_dataset(nc_file)[var_name]
    elif isinstance(nc_file, xr.Dataset):
        nc = nc_file[var_name]
    else:
        raise TypeError("Impact must be file path or xr DataSet")
    nc = nc.rename("value")
    stacked = nc.stack(new_dim=spatial_dims).fillna(0)

    # filter by years
    if years is not None:
        stacked = stacked.sel({time_dim: slice(str(years[0]), str(years[-1]))})

    # df_imp = nc.to_dataframe()
    n_ev = len(stacked[time_dim])
    n_years = len(np.unique(stacked[time_dim].dt.year))

    if "lon" in stacked.coords:
        coord_exp = np.array([stacked.lat.values, stacked.lon.values]).T
        crs = "EPSG:4326"
    else:
        raise NotImplementedError("Only lat/lon coordinates are supported")

    imp = Impact(
        event_id=np.arange(n_ev) + 1,
        event_name=np.array(
            [d.strftime("ev_%Y-%m-%d") for d in stacked[time_dim].dt.date.values]
        ),
        date=np.array([d.toordinal() for d in stacked[time_dim].dt.date.values]),
        coord_exp=coord_exp,
        imp_mat=sparse.csr_matrix(stacked),
        crs=crs,
        eai_exp=stacked.sum(dim=[time_dim]).values / n_years,
        at_event=stacked.sum(dim="new_dim").values,
        frequency=np.ones(n_ev) / n_years,
        aai_agg=float(stacked.sum(dim=["new_dim", time_dim]).values) / n_years,
        unit=unit,
    )

    return imp


# load exposure and impacts
def load_exposure_and_impacts(f_damage, variable, thresh_exp, n_days, date_list):

    exp, condition = load_exposure(f_damage, variable, thresh_exp)

    # create 2D array of exposure values
    exp_values = exp.gdf.value.values
    exp_2D = np.tile(exp_values, n_days).reshape(n_days, exp_values.size)
    del exp_values

    # load impacts
    imp2021 = read_xr_impact(f_damage, "n_count").select(
        dates=["2021-04-01", "2021-09-30"]
    )
    imp2022 = read_xr_impact(f_damage, "n_count").select(
        dates=["2022-04-01", "2022-09-30"]
    )
    imp2023 = read_xr_impact(f_damage, "n_count").select(
        dates=["2023-04-02", "2023-09-30"]
    )
    imp2022 = imp2022.select(
        event_ids=np.delete(
            imp2022.event_id,
            np.where([id in [740, 741, 742, 749, 750, 755] for id in imp2022.event_id]),
        )
    )
    imp = Impact.concat([imp2021, imp2022, imp2023])
    del imp2021, imp2022, imp2023

    # get dates of events with reported damages
    date_events = [dt.date.fromordinal(date) for date in imp.date]
    n_events = len(date_events)

    # select subset of centroids
    imp_coord = imp.coord_exp[condition, :]
    imp = imp.imp_mat[:, condition].toarray()

    # assert consistent coordinates for exposure and impact
    np.testing.assert_array_almost_equal(
        np.array([exp.latitude, exp.longitude]).T, imp_coord
    )

    # loop over events with damages and fill them into zero array
    imp_filled = np.zeros((n_days, imp.shape[1]))
    for i in range(n_events):
        # get date of event and perform ugly type conversion (improve!)
        date_event = dt.datetime(
            date_events[i].year, date_events[i].month, date_events[i].day, 0, 0, 0
        )

        # fill imp at index of event_date within date_list
        index = np.where(date_list == date_event)
        imp_filled[index, :] = imp[i]

    # replace imp_sel object
    imp = imp_filled
    del imp_filled

    return exp, exp_2D, imp, imp_coord


# get icon paths
def get_icon_paths(date_list):
    def shift(i):
        if i % 3 == 1:
            return -1
        if i % 3 == 2:
            return 1
        return 0

    mapping = {
        date: date + dt.timedelta(hours=shift(i) * 6)
        for i, date in enumerate(
            pd.Series(index=pd.date_range("2021-04-01", "2023-09-30", freq="D")).index
        )
    }
    if isinstance(date_list, pd.Timestamp):
        date_list = [date_list]
    date_list_mapped = {date: mapping[date] for date in date_list}

    return [
        f"FCST{curr_date.strftime('%y')}/{mapped_date.strftime('%m')}/"
        + f"DHAIL66mx_init{mapped_date.strftime('%y%m%d_%H')}"
        + f"_val{curr_date.strftime('%y%m%d_')}07-{(curr_date + dt.timedelta(days=1)).strftime('%y%m%d_')}06.nc"
        for curr_date, mapped_date in date_list_mapped.items()
    ]


# load hazard
def load_and_preprocess_hazard(dir_cosmo, run, date_list):
    if run == "flexible":
        haz_paths = [f"{dir_cosmo}{path}" for path in get_icon_paths(date_list)]
    else:
        haz_paths = [
            f"{dir_cosmo}FCST{curr_date.strftime('%y')}/{curr_date.strftime('%m')}/"
            + f"DHAIL66mx_init{curr_date.strftime('%y%m%d_')}{str(run).zfill(2)}"
            + f"_val{curr_date.strftime('%y%m%d_')}{str(run+1).zfill(2)}-{(curr_date + dt.timedelta(days=1)).strftime('%y%m%d_')}{str(run).zfill(2)}.nc"
            for curr_date in date_list
        ]
    ds_haz = xr.open_mfdataset(
        haz_paths, concat_dim="time", combine="nested", coords="minimal"
    )

    haz = hazard_from_radar(
        ds_haz,
        varname="DHAIL_MX",
        time_dim="time",
        ensemble_dim="epsd_1",
        spatial_dims=["x_1", "y_1"],
    )

    return haz


def hazard_from_radar(
    files,
    varname="MESHS",
    time_dim="time",
    forecast_init=None,
    ensemble_dim=None,
    spatial_dims=None,
    country_code=None,
    extent=None,
    subdaily=False,
    month=None,
    ignore_date=False,
    n_year_input=None,
    get_xarray=False,
):
    """Create a new Hail hazard from MeteoCH radar data
    or COSMO HAILCAST ouput (single- or multi-member)

    Parameters
    ----------
    files : list of str or xarray Dataset
        list of netcdf filenames (string) or xarray Dataset object
    varname : string
        the netcdf variable name to be read from the file
    time_dim : str
        Name of time dimension, default: 'time'
    forecast_init : datetime object
        List with datetimes of forecast initializations,
        needs to have same length as time, default: None
    ensemble_dim : str
        Name of ensemble dimension, default: None
    spatial_dims : list of str
        Names of spatial dimensions
    country_code : int
        ISO 3166 country code to filter the data
    extent : list / array
        [lon_min, lon_max, lat_min, lat_max]
    ignore_date : boolean
        If True: ignores netcdf dates (e.g. for synthetic data).
    n_year_input : int
        Number of years: will only be used if ignore_date=True
    Returns
    -------
    haz : Hazard object
        Hazard object containing radar data with hail intensity (MESHS)
    """

    # Initialize default values
    if spatial_dims is None:
        spatial_dims = ["chy", "chx"]

    # read netcdf if it is given as a path
    if type(files) == xr.core.dataset.Dataset:
        netcdf = files
    else:
        netcdf = xr.open_mfdataset(
            files, concat_dim=time_dim, combine="nested", coords="minimal"
        )

    # select month of the year if given
    if month:
        grouped = netcdf.groupby("time.month")
        netcdf = grouped[int(month)]

    # Cut data to selected country/area only
    if extent:
        lon_min, lon_max, lat_min, lat_max = extent
        lon_cond = np.logical_and(netcdf.lon >= lon_min, netcdf.lon <= lon_max)
        lat_cond = np.logical_and(netcdf.lat >= lat_min, netcdf.lat <= lat_max)
        cond = np.logical_and(lat_cond, lon_cond).compute()
        netcdf = netcdf.where(cond, drop=True)

    # Select variable and set units
    varname_xr = varname  # by default the varname corresponds to the xr name
    if varname in ["MESHS", "MESHS_4km", "MESHS_opt_corr", "MESHS_20to23"]:
        varname_xr = "MZC"
        unit = "mm"
    elif varname == "MESHSdBZ" or varname == "MESHSdBZ_p3":
        varname_xr = "MESHSdBZ"
        unit = "mm"
    elif varname == "POH":
        varname_xr = "BZC"
        unit = "%"
    elif "DHAIL" in varname:
        unit = "mm"
    elif varname == "MESH":
        # calculate MESH from SHI
        netcdf = 2.54 * (netcdf) ** 0.5  # Witt et al 1998
        netcdf = netcdf.rename_vars({"SHI": "MESH"})
        # round to steps of 0.5mm
        netcdf["MESH"] = (
            np.round(netcdf["MESH"] / 0.5) * 0.5
        )  # round to steps of 0.5mm (for calibration)
        netcdf["MESH"] = netcdf["MESH"].where(
            netcdf["MESH"] < 80, 0
        )  # outlier values to zero
        unit = "mm"

    elif varname == "dBZ" or varname == "dBZfiltered":
        varname_xr = "CZC"
        unit = "dBZ"
        # Filter values for efficient calculation. dBZ<40 are set to zero
        netcdf = netcdf.where(netcdf[varname_xr] > 40, 0)
    elif varname == "possible_hail":
        unit = "[ ](boolean)"
    elif varname == "durPOH":
        varname_xr = "BZC80_dur"
        netcdf[varname_xr] = netcdf[varname_xr] * 5  # times 5 to get minutes
        unit = "[min]"
    elif varname == "MESHSweigh":
        unit = "mm (scaled by duration)"
    elif varname == "HKE":
        unit = "Jm-2"
    elif varname == "crowd" or varname == "crowdFiltered":
        # warnings.warn('use smoothed data for crowd-sourced data')
        varname_xr = "h_smooth"
        unit = "mm"
    elif (
        varname == "E_kin" or varname == "E_kinCC"
    ):  # E_kin from Waldvogel 1978, or Cecchini 2022
        varname_xr = "E_kin"
        unit = "Jm-2"
    elif varname == "VIL":
        unit = "g/m2"
        varname_xr = "dLZC"
        # Filter values for efficient calculation. VIL<10g/m2 are set to zero
        netcdf = netcdf.where(netcdf[varname_xr] > 10, 0).round(0)
    else:
        raise ValueError(f'varname "{varname}" is not implemented at the moment')

    # prepare xarray with ensemble dimension to be read as climada Hazard
    if ensemble_dim:
        # omit extent if ensemble_dim is given
        if extent:
            warnings.warn(
                "Do not use keyword extent in combination with "
                "ensemble_dim. Plotting will not work."
            )
        # omit igonore_date if ensemble_dim is given
        if ignore_date:
            warnings.warn(
                "Do not use keyword ignore_date in combination with "
                "ensemble_dim. Event names are set differently."
            )
        # stack ensembles along new dimension
        netcdf = netcdf.stack(time_ensemble=(time_dim, ensemble_dim))

        # event names
        if forecast_init:  # event_name = ev_YYMMDD_ensXX_init_YYMMDD_HH
            (n_member,) = np.unique(netcdf[ensemble_dim]).shape
            forecast_init = np.repeat(forecast_init, n_member)
            if netcdf[time_dim].size != len(forecast_init):
                warnings.warn("Length of forecast_init doesn't match time.")
            event_name = np.array(
                [
                    f"{pd.to_datetime(ts).strftime('ev_%y%m%d')}_ens{ens:02d}_{init.strftime('init_%y%m%d_%H')}"
                    for (ts, ens), init in zip(
                        netcdf.time_ensemble.values, forecast_init
                    )
                ]
            )
        else:  # event_name = ev_YYMMDD_ensXX
            event_name = np.array(
                [
                    f"{pd.to_datetime(ts).strftime('ev_%y%m%d')}_ens{ens:02d}"
                    for ts, ens in netcdf.time_ensemble.values
                ]
            )
        # convert MultiIndex to SingleIndex
        netcdf = netcdf.reset_index("time_ensemble")
        netcdf = netcdf.assign_coords({"time_ensemble": netcdf.time_ensemble.values})

        if "time_ensemble" in netcdf.lat.dims:
            # remove duplicates along new dimension for variables that are identical across members (only if lat/lon are saved as variables rather than coordinates)
            netcdf["lon"] = netcdf["lon"].sel(time_ensemble=0, drop=True)
            netcdf["lat"] = netcdf["lat"].sel(time_ensemble=0, drop=True)

    # get number of events and create event ids
    n_ev = netcdf[time_dim].size
    event_id = np.arange(1, n_ev + 1, dtype=int)

    if ignore_date:
        n_years = n_year_input
        if "year" in netcdf.coords:
            event_name = np.array(
                ["ev_%d_y%d" % i for i in zip(event_id, netcdf.year.values)]
            )
        else:
            event_name = np.array(["ev_%d" % i for i in event_id])
    elif ensemble_dim:
        n_years = (
            netcdf[time_dim].dt.year.max().values
            - netcdf[time_dim].dt.year.min().values
            + 1
        )
    else:
        n_years = (
            netcdf[time_dim].dt.year.max().values
            - netcdf[time_dim].dt.year.min().values
            + 1
        )
        if subdaily:
            event_name = netcdf[time_dim].dt.strftime("ev_%Y-%m-%d_%H:%M").values
        else:
            event_name = netcdf[time_dim].dt.strftime("ev_%Y-%m-%d").values

    # Create Hazard object
    event_dim = time_dim
    coord_vars = dict(event=event_dim, longitude="lon", latitude="lat")
    haz = Hazard.from_xarray_raster(
        netcdf, "HL", unit, intensity=varname_xr, coordinate_vars=coord_vars
    )
    # set correct event_name, frequency, date
    haz.event_name = event_name
    haz.frequency = np.ones(n_ev) / n_years
    if ignore_date:
        haz.date = np.array([], int)
    if ensemble_dim:
        haz.date = np.array(
            [pd.to_datetime(ts).toordinal() for ts in netcdf[time_dim].values]
        )

    netcdf.close()
    haz.check()

    if get_xarray:
        return haz, netcdf
    else:
        return haz


# ##############################################
# ### util functions for warnings and skills ###
# ##############################################


def warning_regions_above_exp_threshold(exp_threshold, warning_shape, exp):
    n_exposure_per_warning_region = gpd.sjoin(
        warning_shape.to_crs(ccrs.epsg(21781)),
        exp.gdf.to_crs(ccrs.epsg(21781)),
        how="inner",
        predicate="intersects",
    )
    n_exposure_per_warning_region = n_exposure_per_warning_region.groupby("REGION_NAM")[
        "value"
    ].sum()
    return n_exposure_per_warning_region[
        n_exposure_per_warning_region.values > exp_threshold
    ].index.values


def sum_gdf_to_regions(regions_gdf, gdf):
    joined_gdf_detailed = gpd.sjoin(
        regions_gdf.to_crs(epsg=2056),
        gdf.to_crs(epsg=2056),
        how="inner",
        predicate="intersects",
    )
    return joined_gdf_detailed.groupby("REGION_NAM")[gdf.columns[:-1]].sum()


def df_to_boolean(gdf):
    return gdf[gdf.columns] > 0


def max_gdf_to_regions(regions_gdf, gdf):
    joined_gdf_detailed = gpd.sjoin(
        regions_gdf.to_crs(epsg=2056),
        gdf.to_crs(epsg=2056),
        how="inner",
        predicate="intersects",
    )
    return joined_gdf_detailed.groupby("REGION_NAM")[gdf.columns[:-1]].max()


# def count_gdf_to_regions(
#       regions_gdf,
#       gdf
# ):
#     joined_gdf_detailed = gpd.sjoin(regions_gdf.to_crs(epsg=2056), gdf.to_crs(epsg=2056), how="inner", predicate="intersects")
#     return joined_gdf_detailed.groupby("REGION_NAM")[gdf.columns[:-1]].count()


def create_warning_map_from_gdf(warning_in_regions, regions_shape, date):
    result = regions_shape.copy()
    result["value"] = np.NaN
    for region_name in warning_in_regions[[date]].index:
        result.loc[result["REGION_NAM"] == region_name, "value"] = (
            warning_in_regions[[date]].loc[region_name].values[0]
        )
    return result


def compute_boolean_skill_scores_per_region(df_observed, df_modelled):
    df_index = df_observed.index
    if isinstance(df_observed, pd.DataFrame):
        df_observed = df_observed.values
    if isinstance(df_modelled, pd.DataFrame):
        df_modelled = df_modelled.values

    POD = np.sum(df_observed & df_modelled) / np.sum(df_observed)
    POD_regions = pd.DataFrame(
        np.sum(df_observed & df_modelled, axis=1) / np.sum(df_observed, axis=1),
        index=df_index,
    )

    FAR = np.sum(~df_observed & df_modelled) / np.sum(df_modelled)
    FAR_regions = pd.DataFrame(
        np.sum(~df_observed & df_modelled, axis=1) / np.sum(df_modelled, axis=1),
        index=df_index,
    )

    # FPR = np.sum(~df_observed & df_modelled)/np.sum(~df_observed)
    # FPR_regions = pd.DataFrame(
    #     np.sum(~df_observed & df_modelled, axis=1)/np.sum(~df_observed, axis=1),
    #     index=df_index)

    # rate_TP = np.sum(df_observed & df_modelled)/np.sum(~df_observed | df_observed)
    # rate_TP_regions = pd.DataFrame(
    #     np.sum(df_observed & df_modelled, axis=1)/np.sum(~df_observed | df_observed, axis=1),
    #     index=df_index)

    # rate_TN = np.sum(~df_observed & ~df_modelled)/np.sum(~df_observed | df_observed)
    # rate_TN_regions = pd.DataFrame(
    #     np.sum(~df_observed & ~df_modelled, axis=1)/np.sum(~df_observed | df_observed, axis=1),
    #     index=df_index)

    # rate_FP = np.sum(~df_observed & df_modelled)/np.sum(~df_observed | df_observed)
    # rate_FP_regions = pd.DataFrame(
    #     np.sum(~df_observed & df_modelled, axis=1)/np.sum(~df_observed | df_observed, axis=1),
    #     index=df_index)

    # rate_FN = np.sum(df_observed & ~df_modelled)/np.sum(~df_observed | df_observed)
    # rate_FN_regions = pd.DataFrame(
    #     np.sum(df_observed & ~df_modelled, axis=1)/np.sum(~df_observed | df_observed, axis=1),
    #     index=df_index)

    return {
        "POD": POD,
        "POD_regions": POD_regions,
        "FAR": FAR,
        "FAR_regions": FAR_regions,
        # "FPR": FPR,
        # "FPR_regions": FPR_regions,
        # "TP": rate_TP,
        # "TP_regions": rate_TP_regions,
        # "TN": rate_TN,
        # "TN_regions": rate_TN_regions,
        # "FP": rate_FP,
        # "FP_regions": rate_FP_regions,
        # "FN": rate_FN,
        # "FN_regions": rate_FN_regions,
    }


def POD_weighted(df_obs_complete, df_modelled_boolean):

    if isinstance(df_obs_complete, pd.DataFrame):
        df_obs_complete = df_obs_complete.values
    if isinstance(df_modelled_boolean, pd.DataFrame):
        df_modelled_boolean = df_modelled_boolean.values

    return np.sum(df_obs_complete * df_modelled_boolean) / np.sum(df_obs_complete)


def FAR_weighted(df_impacts, df_modelled_boolean, df_exposure):

    if not isinstance(df_impacts, pd.DataFrame):
        raise TypeError("impacts not gdf")
    if not isinstance(df_modelled_boolean, pd.DataFrame):
        raise TypeError("warnings not gdf")
    if not isinstance(df_exposure, pd.DataFrame):
        raise TypeError("exposure not gdf")

    df_combined = df_modelled_boolean.astype(float)
    df_combined["exposure"] = df_exposure
    df_combined.iloc[:, :-1] = df_combined.iloc[:, :-1].multiply(
        df_combined["exposure"], axis=0
    )
    df_combined.drop(columns=["exposure"], inplace=True)

    falsely_warned_population = np.clip(df_combined - df_impacts, 0, None)
    return np.sum(falsely_warned_population.values) / np.sum(df_combined.values)


# def FPR_weighted(df_impacts, df_modelled_boolean, df_exposure):

#     if not isinstance(df_impacts, pd.DataFrame):
#         raise TypeError("impacts not df")
#     if not isinstance(df_modelled_boolean, pd.DataFrame):
#         raise TypeError("warnings not df")
#     if not isinstance(df_exposure, pd.DataFrame):
#         raise TypeError("exposure df")


#     # calculate how many people were warned on a given date and warn region
#     df_combined = df_modelled_boolean.astype(float)
#     df_combined["exposure"] = df_exposure
#     df_combined.iloc[:,:-1] = df_combined.iloc[:,:-1].multiply(df_combined['exposure'], axis=0)
#     df_combined.drop(columns=["exposure"], inplace=True)
#     # df_combined = df_combined.multiply(df_exposure, axis=0)

#     # calculate how many people were not affected on a given date and warn region
#     df_impacts_boolean = df_to_boolean(df_impacts)
#     df_noocurrence = (~df_impacts_boolean).astype(float)
#     df_noocurrence["exposure"] = df_exposure
#     df_noocurrence.iloc[:,:-1] = df_noocurrence.iloc[:,:-1].multiply(df_noocurrence['exposure'], axis=0)
#     df_noocurrence.drop(columns=["exposure"], inplace=True)
#     # df_noocurrence = df_noocurrence.multiply(df_exposure, axis=0)

#     falsely_warned_population = np.clip(df_combined - df_impacts,0,None)
#     return np.sum(falsely_warned_population.values)/np.sum(df_noocurrence.values)


# Emanuel impact function
def get_emanuel_impf(
    v_thresh=20,
    v_half=60,
    scale=1e-3,
    power=3,
    impf_id=1,
    intensity=np.arange(0, 110, 1),
    intensity_unit="mm",
    haz_type="HL",
):
    """
    Init TC impact function using the formula of Kerry Emanuel, 2011:
    https://doi.org/10.1175/WCAS-D-11-00007.1

    Parameters
    ----------
    impf_id : int, optional
        impact function id. Default: 1
    intensity : np.array, optional
        intensity array in m/s. Default:
        5 m/s step array from 0 to 120m/s
    v_thresh : float, optional
        first shape parameter, wind speed in
        m/s below which there is no damage. Default: 25.7(Emanuel 2011)
    v_half : float, optional
        second shape parameter, wind speed in m/s
        at which 50% of max. damage is expected. Default:
        v_threshold + 49 m/s (mean value of Sealy & Strobl 2017)
    scale : float, optional
        scale parameter, linear scaling of MDD.
        0<=scale<=1. Default: 1.0
    power : int, optional
        Exponential dependence. Default to 3 (as in Emanuel (2011))

    Raises
    ------
    ValueError

    Returns
    -------
    impf : ImpfTropCyclone
        TC impact function instance based on formula by Emanuel (2011)
    """
    if v_half <= v_thresh:
        raise ValueError("Shape parameters out of range: v_half <= v_thresh.")
    if v_thresh < 0 or v_half < 0:
        raise ValueError("Negative shape parameter.")
    if scale > 1 or scale <= 0:
        raise ValueError("Scale parameter out of range.")

    impf = ImpactFunc(
        haz_type=haz_type,
        id=impf_id,
        intensity=intensity,
        intensity_unit=intensity_unit,
        name="Emanuel-type",
    )
    impf.paa = np.ones(intensity.shape)
    v_temp = (impf.intensity - v_thresh) / (v_half - v_thresh)
    v_temp[v_temp < 0] = 0
    impf.mdd = v_temp**power / (1 + v_temp**power)
    impf.mdd *= scale
    return impf


def select_large_forecasts(
    estimated_values,
    date_list,
    OOM_threhsold,
    q_selection,
    reported_values=None,
    n_members=33,
):
    ind_median_above_threshold = (
        np.quantile(estimated_values[:, :n_members], q_selection, axis=1)
        >= OOM_threhsold
    )
    estimated_large = estimated_values[ind_median_above_threshold]
    dates_large = date_list[ind_median_above_threshold]
    if reported_values is not None:
        reported_large = reported_values[ind_median_above_threshold]
    else:
        reported_large = None
    return (estimated_large, dates_large, reported_large)


def ensemble_range(estimated_values, q_min, q_max, threshold_values=None, n_members=33):
    if threshold_values:
        estimated_values = np.clip(estimated_values, a_min=threshold_values, a_max=None)
    return (
        np.log10(np.quantile(estimated_values[:, :n_members], q_max, axis=1)),
        np.log10(np.quantile(estimated_values[:, :n_members], q_min, axis=1)),
    )


def ensemble_spread_OOM(
    estimated_values,
    q_min,
    q_max,
    threshold_values=None,
    n_members=33,
):
    if threshold_values:
        estimated_values = np.clip(estimated_values, a_min=threshold_values, a_max=None)
    return np.log10(
        np.quantile(estimated_values[:, :n_members], q_max, axis=1)
    ) - np.log10(np.quantile(estimated_values[:, :n_members], q_min, axis=1))


def rate_reported_in_qunatile_range(
    dmgs_modelled,
    dmgs_observed,
    qmin,
    qmax,
    min_threshold_on_observed=0,
    n_members=33,
    exp_modelled=None,
    exp_observed=None,
):
    if exp_modelled is not None and exp_observed is not None:
        n_events = int(len(dmgs_observed) / len(exp_observed))
        exp_observed = np.concatenate([exp_observed] * n_events)
        exp_modelled = np.concatenate([exp_modelled] * n_events)
        rel_dmgs_observed = (dmgs_observed.T / exp_observed).T
        rel_dmgs_modelled = (dmgs_modelled.T / exp_modelled).T
    else:
        rel_dmgs_observed = dmgs_observed
        rel_dmgs_modelled = dmgs_modelled

    return np.sum(
        (
            rel_dmgs_observed
            <= np.quantile(rel_dmgs_modelled[:, :n_members], qmax, axis=1)
        )
        & (
            rel_dmgs_observed
            >= np.quantile(rel_dmgs_modelled[:, :n_members], qmin, axis=1)
        )
        & (dmgs_observed >= min_threshold_on_observed)
    ) / np.sum((dmgs_observed >= min_threshold_on_observed))


def rate_quantile_above_1(
    dmgs_modelled,
    dmgs_observed,
    q,
    max_threshold_on_observed=1,
    n_members=33,
):
    return np.sum(
        (np.quantile(dmgs_modelled[:, :n_members], q, axis=1) >= 1)
        & (dmgs_observed < max_threshold_on_observed)
    ) / np.sum((dmgs_observed < max_threshold_on_observed))
