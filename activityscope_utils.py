"""
ActivitySCOPE Utilities Module

This module contains utility functions for:
- Loading orbital databases (MPC, AstDyS, JPL)
- Loading astrometry counts
- Comparing databases
- Feature engineering for machine learning models
- Model hyperparameters and scoring functions
- Extension difficulty classification
"""

import os
import shutil
import time
import warnings
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from sbpy.data import Names
import json
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import cross_val_predict
from autogluon.core.metrics import make_scorer
from xgboost import XGBClassifier


# ==============================================================================
# IAU H-G PHASE FUNCTION (G = 0.15)
# ==============================================================================
# Shared helper for the phase-angle dimming term, -2.5*log10(Phi(alpha)), used
# by vis_inc; the same blended HG phase function is also inlined in the
# orbit-averaged feature loops. Returns the blended phase factor Phi(alpha) in
# [0, 1]; alpha is the phase angle in radians.
_HG_A1, _HG_B1 = 3.33, 0.63
_HG_A2, _HG_B2 = 1.87, 1.22
_HG_G = 0.15
_PHI_FLOOR = 1e-30


def hg_phase(alpha_rad):
    """Blended IAU H-G phase function Phi(alpha) for G = 0.15."""
    tan_half = np.maximum(np.tan(np.clip(alpha_rad, 0.0, np.pi) / 2.0), 0.0)
    phi1 = np.exp(-_HG_A1 * np.power(tan_half, _HG_B1))
    phi2 = np.exp(-_HG_A2 * np.power(tan_half, _HG_B2))
    return np.maximum((1.0 - _HG_G) * phi1 + _HG_G * phi2, _PHI_FLOOR)


# ==============================================================================
# MPC ORBIT DOWNLOAD CACHE
# ==============================================================================

_MPCORB_URL = "https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz"
_MPCORB_CACHE_PATH = ".cache/mpcorb_extended.json.gz"
_MPCORB_CACHE_TTL_SECONDS = 60*20  # 20 minutes


def get_mpcorb_extended_path(max_age_seconds=_MPCORB_CACHE_TTL_SECONDS):
    """
    Return a local path to mpcorb_extended.json.gz, downloading it to a local
    cache on first use and reusing it on subsequent calls within the TTL.
    Pass max_age_seconds=0 to force a refresh.
    """
    cache_dir = os.path.dirname(_MPCORB_CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    fresh = (
        os.path.exists(_MPCORB_CACHE_PATH)
        and (time.time() - os.path.getmtime(_MPCORB_CACHE_PATH)) < max_age_seconds
    )
    if not fresh:
        tmp_path = _MPCORB_CACHE_PATH + ".part"
        if os.path.exists(_MPCORB_CACHE_PATH):
            os.remove(_MPCORB_CACHE_PATH)

        req = urllib.request.Request(
            _MPCORB_URL,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"},
        )
        with urllib.request.urlopen(req) as r, open(tmp_path, "wb") as f:
            shutil.copyfileobj(r, f)

        os.replace(tmp_path, _MPCORB_CACHE_PATH)
    return _MPCORB_CACHE_PATH


# ==============================================================================
# MODEL HYPERPARAMETERS AND SCORING
# ==============================================================================

from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import copy

def _get_hyperparameters_lgbm_xgb_only():
    """Fetches the default AutoGluon hyperparameters and keeps only the two
    gradient-boosted tree algorithms used throughout ActivitySCOPE: LightGBM
    ('GBM') and XGBoost ('XGB'). All other default models (CatBoost, neural
    networks, random forest, extra trees, KNN, etc.) are dropped."""
    hp = copy.deepcopy(get_hyperparameter_config('default'))
    return {k: hp[k] for k in ('GBM', 'XGB') if k in hp}

# Hyperparameters for binary classification models
HYPERPARAMETERS_BINARY = {
    "GBM": [
        {},
        {
            "learning_rate": 0.03,
            "num_leaves": 128,
            "feature_fraction": 0.9,
            "min_data_in_leaf": 3,
            "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
        },
    ],
    "XGB": [{}, {"learning_rate": 0.5, "max_depth": 3, "min_child_weight": 18, "subsample": 1.0, "ag_args": {"name_suffix": "_tuned"}}]
}

# Hyperparameters for Poisson regression models
HYPERPARAMETERS_POISSON = {
    'GBM': {'objective': 'poisson', 'num_iterations': 1000, 'learning_rate': 0.1},
    'XGB': {'objective': 'count:poisson'},
}

# Hyperparameters for Quantile regression models
HYPERPARAMETERS_QUANTILE = _get_hyperparameters_lgbm_xgb_only()

# Custom Poisson scorer for regression model evaluation
POISSON_SCORER = make_scorer(
    name='mean_poisson_deviance',
    score_func=mean_poisson_deviance,
    optimum=0,
    greater_is_better=False
)


# ==============================================================================
# ORBITAL DATABASE LOADING
# ==============================================================================

def load_mpc_orbits():
    """
    Load the MPC orbit database and apply initial processing.
    
    Returns
    -------
    pd.DataFrame
        Processed MPC orbit database
    """
    orb = pd.read_json(get_mpcorb_extended_path(), compression='gzip')

    # MPC leaves the U (orbit-uncertainty) parameter blank for some older objects
    # deemed lost. We fill it with 10 (worse than the U=9 maximum), even though a
    # few objects with a missing U actually have a better-defined orbit than U=9.
    orb['U'] = pd.to_numeric(orb['U'], errors='coerce').fillna(10)
    
    orb = orb.convert_dtypes()
    orb.drop(["Other_desigs"], axis=1, inplace=True)
    
    # Filter based on filter lists
    filter_until_further_notice = pd.read_csv("filter until further notice.csv")
    filter_out_unless_updated = pd.read_csv("filter out unless updated.csv")
    
    # We will create a column that marks it as filtered_out
    orb["filtered_out"] = orb["Principal_desig"].isin(filter_until_further_notice["Object"]).astype(int)
    
    for _, row in filter_out_unless_updated.iterrows():
        mask = (orb["Principal_desig"] == row["Object"]) & (orb["Arc_length"] == row["Arc_length"])
        orb.loc[mask, "filtered_out"] = 1
    
    return orb


def load_astrometry_counts(orb):
    """
    Load astrometry counts and merge with orbit dataframe.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe to merge with astrometry counts
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with astrometry counts merged
    """
    with open('./astrometry_counter/astrometry_counts.json', 'r') as f:
        astrometry_counts = json.load(f)
    
    astrometry_counts_df = pd.DataFrame.from_dict(astrometry_counts, orient='index')
    astrometry_counts_df.index = astrometry_counts_df.index.map(
        lambda x: Names.from_packed(x).format("desig")
    )
    astrometry_counts_df.index.name = "Principal_desig"
    astrometry_counts_df["other_opps"] = (
        astrometry_counts_df["nights_total"] - 
        astrometry_counts_df["opp_with_most_nights"]
    )
    
    orb = orb.merge(astrometry_counts_df, how="left", 
                   left_on="Principal_desig", right_index=True)
    
    return orb


def load_astdys_orbits():
    """
    Load AstDyS orbit database (multi-opposition and single-opposition).
    
    Returns
    -------
    pd.DataFrame
        AstDyS orbit database
    """
    astdys_names = ["Astdys Name", "Epoch-MJD", "a", "e", "i", "Node", "Peri", 
                    "M", "H", "G", "rand"]
    astdys_widths = [15, 12, 25, 25, 25, 25, 25, 25, 6, 6, 3]
    
    # Multi-opposition objects
    astdys = pd.read_fwf(
        "https://newton.spacedys.com/~astdys2/catalogs/ufitobs.cat",
        index_col=False, names=astdys_names, widths=astdys_widths, skiprows=6
    )
    astdys["Astdys Multiopp"] = 1
    
    # Single-opposition objects
    astdys_sing = pd.read_fwf(
        "https://newton.spacedys.com/~astdys2/catalogs/singopp.cat",
        index_col=False, names=astdys_names, widths=astdys_widths, skiprows=6
    )
    astdys_sing["Astdys Multiopp"] = 0
    
    # Combine
    astdys = pd.concat([astdys, astdys_sing])
    print(f"Loaded {len(astdys)} AstDyS orbits")
    
    astdys = astdys.convert_dtypes()
    astdys["n"] = 360 / (astdys["a"])**1.5 / 365.2569
    astdys["Perihelion_dist"] = astdys["a"] * (1 - astdys["e"])
    astdys["Aphelion_dist"] = astdys["a"] * (1 + astdys["e"])
    astdys["Epoch"] = astdys["Epoch-MJD"] + 2400000.5
    astdys["Astdys Name"] = astdys["Astdys Name"].str.replace("'", "")
    astdys["Astdys Name"] = (astdys["Astdys Name"].str.slice(0, 4) + " " + 
                             astdys["Astdys Name"].str.slice(4))
    astdys["Ref"] = "AstDyS"
    
    return astdys


def load_jpl_orbits():
    """
    Load JPL orbit database.
    
    Returns
    -------
    pd.DataFrame
        JPL orbit database
    """
    jpl_names = ["Desig", "Epoch-MJD", "a", "e", "i", "Peri", "Node", "M", 
                 "H", "G", "Ref"]
    jpl_widths = [14, 6, 12, 11, 10, 10, 10, 12, 6, 5, 10]
    
    jpl = pd.read_fwf(
        "https://ssd.jpl.nasa.gov/dat/ELEMENTS.UNNUM.gz",
        compression='gzip', index_col=False, names=jpl_names, 
        widths=jpl_widths, skiprows=2
    )
    jpl = jpl.convert_dtypes()
    
    return jpl


# ==============================================================================
# DATABASE COMPARISON
# ==============================================================================

def compare_with_astdys(orb, astdys):
    """
    Compare MPC orbits with AstDyS and add comparison metrics.
    
    Parameters
    ----------
    orb : pd.DataFrame
        MPC orbit dataframe
    astdys : pd.DataFrame
        AstDyS orbit dataframe
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with AstDyS comparison columns added
    """
    orb = orb.merge(
        astdys[["Astdys Name", "a", "e", "i", "H", "Astdys Multiopp"]], 
        how="left", 
        left_on="Principal_desig", 
        right_on="Astdys Name", 
        suffixes=("", "_astdys")
    )
    
    # Set multi_opp_disagree to 1 any time Astdys Multiopp is defined and 
    # disagrees with Num_opps
    mpc_multiopp = (orb["Num_opps"] > 1).astype(int)
    orb["multi_opp_disagree"] = (
        (orb["Astdys Multiopp"].notna()) & 
        (orb["Astdys Multiopp"] != mpc_multiopp)
    ).astype(int)
    
    return orb


def compare_with_jpl(orb, jpl):
    """
    Compare MPC orbits with JPL and add comparison metrics.
    
    Parameters
    ----------
    orb : pd.DataFrame
        MPC orbit dataframe
    jpl : pd.DataFrame
        JPL orbit dataframe
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with JPL comparison columns added
    """
    orb = orb.merge(
        jpl[["Desig", "H"]], 
        how="left", 
        left_on="Principal_desig", 
        right_on="Desig", 
        suffixes=("", "_jpl")
    )
    
    return orb


def compute_database_differences(orb):
    """
    Compute differences between MPC, AstDyS, and JPL orbital elements.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe with MPC, AstDyS, and JPL data
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with difference columns added
    """
    # For any H above 33, make it into an NA
    orb.loc[orb['H'] > 33, 'H'] = pd.NA
    orb['H_MPC'] = orb['H']
    orb.loc[orb['H_astdys'] > 33, 'H_astdys'] = pd.NA
    orb.loc[orb['H_jpl'] > 33, 'H_jpl'] = pd.NA
    
    # Figure out difference between H and H_astdys
    orb["H_diff_abs"] = (orb["H"] - orb["H_astdys"]).abs()
    orb["a_diff_abs"] = (orb["a"] - orb["a_astdys"]).abs()
    orb["e_diff_abs"] = (orb["e"] - orb["e_astdys"]).abs()
    orb["i_diff_abs"] = (orb["i"] - orb["i_astdys"]).abs()
    
    # Fill NAs with small default values
    orb["H_diff_abs"] = orb["H_diff_abs"].fillna(0.011)
    orb["a_diff_abs"] = orb["a_diff_abs"].fillna(0)
    orb["e_diff_abs"] = orb["e_diff_abs"].fillna(0)
    orb["i_diff_abs"] = orb["i_diff_abs"].fillna(0)
    
    # For JPL, we will only compare H for now as that's the thing we most need 
    # to be certain of
    orb["H_diff_abs_jpl"] = (orb["H"] - orb["H_jpl"]).abs()
    orb["H_diff_abs_jpl"] = orb["H_diff_abs_jpl"].fillna(0.012)
    
    orb["H_diff_abs_max"] = orb[["H_diff_abs", "H_diff_abs_jpl"]].max(axis=1)
    
    return orb


def apply_magnitude_corrections(orb, corrections_file="absolute magnitude fixes.csv"):
    """
    Apply corrected H magnitudes from known photometry issues.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe
    corrections_file : str, optional
        Path to corrections CSV file
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with corrections applied
    """
    corrections = pd.read_csv(corrections_file)
    for _, row in corrections.iterrows():
        orb.loc[orb["Principal_desig"] == row["Object"], "H"] = row["Corrected H"]
    
    return orb


def apply_nights_overrides(orb, overrides_file="nights_override.csv"):
    """
    Apply overrides for nights_total when the CSV specifies a higher value.

    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe
    overrides_file : str, optional
        Path to overrides CSV file

    Returns
    -------
    pd.DataFrame
        Orbit dataframe with overrides applied
    """
    try:
        overrides = pd.read_csv(overrides_file, skipinitialspace=True)
        for _, row in overrides.iterrows():
            mask = orb["Principal_desig"] == row["Object"]
            if mask.any():
                current_nights = orb.loc[mask, "nights_total"].values[0]
                if pd.isna(current_nights) or row["Min Nights"] > current_nights:
                    orb.loc[mask, "nights_total"] = row["Min Nights"]
    except FileNotFoundError:
        pass

    return orb


def apply_num_opps_overrides(orb, overrides_file="num_opps_overrides.csv"):
    """
    Apply overrides for number of oppositions when the CSV specifies a higher value.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe
    overrides_file : str, optional
        Path to overrides CSV file
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with overrides applied
    """
    try:
        overrides = pd.read_csv(overrides_file, skipinitialspace=True)
        for _, row in overrides.iterrows():
            mask = orb["Principal_desig"] == row["Object"]
            if mask.any():
                current_opps = orb.loc[mask, "Num_opps"].values[0]
                if row["Opps"] > current_opps:
                    orb.loc[mask, "Num_opps"] = row["Opps"]
    except FileNotFoundError:
        pass
    
    # Anytime the number of oppositions is >= 2, set Arc_length to NaN
    orb.loc[orb["Num_opps"] >= 2, "Arc_length"] = np.nan
    
    return orb


def add_training_targets(orb, num_opps_threshold=4):
    """
    Add binary classification and regression targets for training.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe
    num_opps_threshold : int, optional
        Threshold for binary classification (default: 4)
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with training target columns added
    """
    # Indicator variable, the training target for binary classification model
    orb["Is_Past_Threshold"] = (orb["Num_opps"] >= num_opps_threshold) * 1
    
    # This is the training target for the regression model as we take the first 
    # opposition as granted (it wouldn't be designated if it hadn't been observed 
    # at least once) then it's the number of additional oppositions beyond the 
    # first one that we are trying to predict, which is just Num_opps - 1
    orb["Num_opps_minus_one"] = orb["Num_opps"] - 1
    
    return orb


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def feature_engineering(orb, extra_features=False):
    """
    Engineer predictive features from orbital elements.
    
    This function adds visibility metrics, orbital dynamics features, and 
    geometric properties to the dataframe.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe with at least columns: a, e, i, H, Peri, Node
    extra_features : bool, default False
        Also compute the exploratory / superseded feature families that the 12-feature model
        does not use -- the orbit-averaged block (vis_orbit_mag_multi,
        spatial_discoverability_fraction, dec_flux_weighted, vis_orbit_flux_*, ...), the
        last-three-perihelia reconstruction, the 17-apparition vis_opp_* solver,
        n_detect_windows, and the survey-era block (E_50yr, E_var, E_full_w03, first_det_year,
        gal_lat_frac_low, ...) apart from its one surviving column, days_since_last_opp.
        Together they are the great majority of this function's runtime (n_detect_windows
        alone is most of it), so they are off by default; see
        _add_orbit_averaged_features, _add_exploratory_features, add_detect_window_features
        and add_survey_era_features. Turning this on reproduces their values unchanged.
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with engineered features added (modifies in place)
    """
    # ============================================================================
    # VISIBILITY FEATURES
    # ============================================================================
    
    a = orb['a']
    e = np.clip(orb['e'], 0, 0.999)
    H = orb['H']
    eps_val = 1e-3
    
    # vis_timeavg    d = 0.9
    # Typical-geometry visibility magnitude, with both distance factors taken as
    # orbit-averages:
    #   r = a(1 + e^2/2) is the time-averaged heliocentric distance: an asteroid
    #       spends more time near aphelion (Kepler's 2nd law), so <r>_t > a.
    #   Delta = sqrt(r^2 - 1) = sqrt((r - 1)(r + 1)) is the geometric mean of the
    #       opposition (r - 1, closest) and conjunction (r + 1, farthest)
    #       geocentric distances. Because magnitude is logarithmic in distance
    #       (the 5 log10 Delta term), this geometric mean is the distance whose
    #       magnitude equals the average of the best- and worst-case observing
    #       geometries -- a "typical visibility distance".
    #   d (AU) is a small empirical calibration: surveys preferentially detect
    #       objects near opposition, so the detection-weighted distance sits below
    #       the symmetric mean; d nudges Delta toward the near end. Fitted, not
    #       derived.
    d_timeavg = 0.9
    r_timeavg = a * (1.0 + e**2 / 2.0)
    delta_geom = np.sqrt(np.maximum(np.abs(r_timeavg**2 - 1.0), eps_val))
    delta_timeavg = np.maximum(delta_geom - d_timeavg, eps_val)
    orb['vis_timeavg'] = (5.0 * np.log10(np.maximum(r_timeavg, eps_val) * delta_timeavg) + H).astype(float)

    # vis_a: exactly vis_q's formula evaluated at r = a instead of r = q -- the idealized
    # opposition magnitude at the mean distance, with Delta = a - 1 au. Earlier versions used the
    # geometric mean of the opposition and conjunction distances, sqrt(a^2 - 1), less an offset of
    # 1 au; that offset had no geometric meaning, and removing it costs nothing measurable
    # (+0.067% Poisson deviance, 95% CI [+0.008%, +0.130%], and -0.030% log-loss, on 1.17M rows
    # held out from a 200k training subsample). The a - 1 form also needs no absolute value, so it
    # stays defined for interior objects the same way vis_q does.
    delta_a = np.maximum(a - 1.0, eps_val)
    orb['vis_a'] = (5.0 * np.log10(np.maximum(a, eps_val) * delta_a) + H).astype(float)

    # vis_typ
    d_vis_typ = 0.51
    r_vis_typ = a * (1.0 + e / 2.0)
    delta_vis_typ = np.maximum(r_vis_typ - d_vis_typ, eps_val)
    orb['vis_typ'] = (5.0 * np.log10(np.maximum(r_vis_typ, eps_val) * delta_vis_typ) + H).astype(float)

    # vis_flux
    d_flux = 1
    r_flux = a * np.power(np.maximum(1.0 - e**2, 0.0), 0.25)
    delta_flux = np.maximum(r_flux - d_flux, eps_val)
    orb['vis_flux'] = (5.0 * np.log10(np.maximum(r_flux, eps_val) * delta_flux) + H).astype(float)

    # ------------------------------------------------------------------------
    # Simpler-math alternatives to vis_flux. Both keep the identical
    # V = 5*log10(r * Delta) + H template and only change the characteristic
    # heliocentric distance r(a, e). Each r is below the semi-major axis a and
    # decreases with e (bright-biased toward close passages), reproducing the
    # qualitative behavior of vis_flux = a*(1 - e^2)^0.25 = sqrt(a*b), but with
    # an elementary, one-line derivation instead of the Keplerian <1/r^2>_t
    # time-average that fixes vis_flux's quarter-power exponent.
    #
    # Delta = r - d uses d = 1 (Earth at 1 AU, opposition geometry) for both,
    # matching vis_flux and introducing no fitted parameter; d can be refit per
    # feature like the siblings (vis_timeavg=0.9, vis_typ=0.51) if
    # desired.
    d_alt = 1.0

    # vis_smin: r = a*sqrt(1 - e^2) = b, the orbit's semi-minor axis, which is
    # exactly the geometric mean of perihelion and aphelion, sqrt(q*Q) =
    # sqrt(a(1-e) * a(1+e)). One line, a textbook orbital quantity, no integral.
    # Same (1 - e^2)^p form as vis_flux with p = 1/2 instead of 1/4, so it bends
    # the same way but a bit harder (brighter) at high e.
    r_smin = a * np.sqrt(np.maximum(1.0 - e**2, 0.0))
    delta_smin = np.maximum(r_smin - d_alt, eps_val)
    orb['vis_smin'] = (5.0 * np.log10(np.maximum(r_smin, eps_val) * delta_smin) + H).astype(float)

    # vis_mid: r = (q + a)/2 = a*(1 - e/2), the midpoint between perihelion and
    # the semi-major (mean) distance -- "halfway between closest approach and
    # average distance." Linear in e (a different functional shape from the
    # (1-e^2) family above), trivially defensible. It sits below a (bright-
    # biased), the mirror of a*(1 + e/2), so it carries oppositely-signed e
    # information relative to the aphelion-side distances.
    r_mid = a * (1.0 - e / 2.0)
    delta_mid = np.maximum(r_mid - d_alt, eps_val)
    orb['vis_mid'] = (5.0 * np.log10(np.maximum(r_mid, eps_val) * delta_mid) + H).astype(float)
    # ------------------------------------------------------------------------

    # vis_q: perihelion viewed at idealized opposition (Delta = q - 1), no fitted
    # constant. Replaces vis_mid in the paper feature set (-0.15% deviance, -0.11%
    # log-loss, paired 5-fold on the full training frame). An earlier version used a
    # tuned offset of 0.8 au instead of 1; it was measurably worse (+0.4% deviance)
    # and has been removed.
    r_vis_q = a * (1.0 - e)
    delta_vis_q = np.maximum(r_vis_q - 1.0, eps_val)
    orb['vis_q'] = (5.0 * np.log10(np.maximum(r_vis_q, eps_val) * delta_vis_q) + H).astype(float)

    # vis_inc_old
    r_t = a * (1.0 + e**2 / 2.0)
    i_rad_temp = np.radians(orb['i'])
    delta_inc = np.sqrt(np.maximum(r_t**2 - 2.0 * r_t * np.cos(i_rad_temp) + 1.0, eps_val))
    orb['vis_inc_old'] = (5.0 * np.log10(np.maximum(r_t, eps_val) * delta_inc) + H).astype(float)

    # vis_inc: vis_inc_old plus a single HG phase-angle dimming term.
    # The paper calls vis_inc_old "a useful-to-the-model approximation, not a physical
    # point in time" -- and unlike vis_opp_mean / vis_orbit_mag_multi it carries no
    # phase correction. The geometry it already defines (asteroid at r_t, Earth at
    # 1 AU, geocentric distance delta_inc) fixes the Sun-asteroid-Earth phase angle
    # by the law of cosines at the asteroid vertex:
    #   cos alpha = (r^2 + Delta^2 - 1) / (2 r Delta) = (r - cos i) / Delta.
    # Adding -2.5*log10(Phi(alpha)) makes it more physical with no new parameter.
    cos_alpha_inc = np.clip(
        (r_t - np.cos(i_rad_temp)) / np.maximum(delta_inc, eps_val), -1.0, 1.0
    )
    alpha_inc = np.arccos(cos_alpha_inc)
    orb['vis_inc'] = (
        5.0 * np.log10(np.maximum(r_t, eps_val) * delta_inc)
        + H - 2.5 * np.log10(hg_phase(alpha_inc))
    ).astype(float)

    # inc_opp_penalty: vis_inc's inclination insight, stripped of H and the
    # absolute magnitude scale so it is orthogonal to the vis* family. It is the
    # dimming (mag, >= 0) that inclination alone costs an object at opposition,
    # relative to a coplanar (i = 0) twin: the extra geocentric distance
    # (delta_inc vs delta0 = sqrt(r_t^2 - 1)) plus the phase-angle floor that the
    # out-of-ecliptic offset forces. ~0 for ecliptic asteroids, growing with i.
    delta0 = np.sqrt(np.maximum(r_t**2 - 1.0, eps_val))
    orb['inc_opp_penalty'] = (
        5.0 * np.log10(delta_inc / delta0) - 2.5 * np.log10(hg_phase(alpha_inc))
    ).astype(float)
    
    # ============================================================================
    # ORBITAL DYNAMICS FEATURES
    # ============================================================================
    
    # Orbital period resonance with Earth
    # Measures how closely the orbital period matches an integer number of years
    # Period is already measured in years (where we get it from)
    orb['orbital_period_sync'] = np.abs(
        orb['Orbital_period'] - np.round(orb['Orbital_period'])
    )
    
    # Tisserand parameter relative to Jupiter
    # Distinguishes dynamical classes (asteroids vs comets, Trojans, etc.)
    orb["TJ"] = (5.203 / orb["a"] + 
                 2 * np.cos(np.radians(orb["i"])) * 
                 np.sqrt(orb["a"] / 5.203 * (1 - orb["e"]**2)))
    
    # Jupiter Trojan classification
    # Objects in 1:1 resonance with Jupiter near L4/L5 Lagrange points
    orb["is_trojan"] = ((orb["a"] > 5.0) & (orb["a"] < 5.4) &
                        (orb["e"] < 0.3) & (orb["i"] < 40)).astype(int)
    
    # ============================================================================
    # GEOMETRIC FEATURES
    # ============================================================================
    
    # Perihelion direction unit vector in heliocentric ecliptic coordinates
    # Captures seasonal visibility patterns based on perihelion orientation
    perihelion_directions = np.array([
        np.cos(np.radians(orb["Node"])) * np.cos(np.radians(orb["Peri"])) - 
        np.sin(np.radians(orb["Node"])) * np.sin(np.radians(orb["Peri"])) * 
        np.cos(np.radians(orb["i"])),
        
        np.sin(np.radians(orb["Node"])) * np.cos(np.radians(orb["Peri"])) + 
        np.cos(np.radians(orb["Node"])) * np.sin(np.radians(orb["Peri"])) * 
        np.cos(np.radians(orb["i"])),
        
        np.sin(np.radians(orb["Peri"])) * np.sin(np.radians(orb["i"]))
    ])
    
    orb["Perihelion_direction_x"] = perihelion_directions[0]
    orb["Perihelion_direction_y"] = perihelion_directions[1]
    orb["Perihelion_direction_z"] = perihelion_directions[2]
    
    # Eccentricity-weighted perihelion vectors
    # For circular orbits (e≈0), perihelion direction is undefined; 
    # weighting by e resolves this
    orb["Perihelion_direction_x_e"] = orb["Perihelion_direction_x"] * orb["e"]
    orb["Perihelion_direction_y_e"] = orb["Perihelion_direction_y"] * orb["e"]
    orb["Perihelion_direction_z_e"] = orb["Perihelion_direction_z"] * orb["e"]

    # Coplanar eccentricity-vector components: e*cos(varpi), e*sin(varpi) with
    # varpi = Node + Peri the longitude of perihelion. This is Perihelion_direction_{x,y}_e
    # with the cos(i) correction dropped; performance is indistinguishable on the
    # full training frame (-0.03% deviance, +0.10% log-loss, neither significant)
    # and the definition is a single line.
    varpi_rad = np.radians(orb["Node"] + orb["Peri"])
    orb["e_cos_varpi"] = orb["e"] * np.cos(varpi_rad)
    orb["e_sin_varpi"] = orb["e"] * np.sin(varpi_rad)
    
    # Declination of perihelion
    # Accounts for northern vs southern hemisphere observational bias
    eps = np.radians(23.44)  # Earth's axial tilt
    i_rad = np.radians(orb['i'])
    node_rad = np.radians(orb['Node'])
    peri_rad = np.radians(orb['Peri'])
    
    sin_dec = (np.sin(i_rad) * np.sin(peri_rad) * np.cos(eps) + 
               (np.cos(peri_rad) * np.sin(node_rad) + 
                np.sin(peri_rad) * np.cos(i_rad) * np.cos(node_rad)) * 
               np.sin(eps))
    orb['dec_perihelion'] = np.degrees(np.arcsin(np.clip(sin_dec, -1.0, 1.0)))
    
    # Galactic plane alignment
    # Angle between orbital plane and galactic plane (affects stellar 
    # background density)
    n_gal = np.array([-0.86767, -0.00041, 0.49717])  # J2000 NGP in J2000 ecliptic coords
    n_ast = [
        np.sin(np.radians(orb["i"])) * np.sin(np.radians(orb["Node"])),
        -np.sin(np.radians(orb["i"])) * np.cos(np.radians(orb["Node"])),
        np.cos(np.radians(orb["i"]))
    ]
    orb["galactic_inc"] = np.degrees(np.arccos(np.clip(
        n_gal[0]*n_ast[0] + n_gal[1]*n_ast[1] + n_gal[2]*n_ast[2], -1.0, 1.0
    )))
    
    # Combined angular elements (exploratory feature)
    orb["node_plus_peri"] = (orb["Node"] + orb["Peri"]) % 360
    
    # The orbit-averaged block and the exploratory / superseded families (last-perihelia and the
    # 17-apparition vis_opp_* block) are all opt-in: the 12-feature model uses none of their
    # columns. See _add_orbit_averaged_features and _add_exploratory_features.
    if extra_features:
        orb = _add_orbit_averaged_features(orb)
        orb = _add_exploratory_features(orb)

    if "Principal_desig" in orb.columns:
        # Palomar-Leiden ("2060 P-L") and Trojan-survey ("4020 T-2")
        # designations have the same shape as a year but are not one, so require
        # a genuine provisional designation: 4-digit year, space, two letters.
        orb["desig_year"] = pd.to_numeric(
            orb["Principal_desig"].astype("string").str.extract(
                r"^((?:1[89]|20)\d{2})\s+[A-Z]{2}", expand=False
            ),
            errors="coerce",
        ).astype(float)
    else:
        orb["desig_year"] = np.nan

    # Every apparition-derived block below wants the same reconstructed apparitions, so they are
    # solved once here and sliced: add_survey_era_features scores apparitions 0..24 and 25..59,
    # add_detectable_apparition_features all 60. Solving once is bit-identical to the separate
    # solves (see _slice_apparitions) and removes a full duplicate of the solver's work.
    G, t_ref = _shared_apparition_solve(orb)
    if extra_features:
        # The survey-era block (E_50yr, E_var, E_full_w03, first_det_year, gal_lat_frac_low, ...)
        # was superseded by the detectable-apparition features below; only days_since_last_opp
        # survived into the model, and add_last_opposition_feature computes that alone.
        orb = add_survey_era_features(orb, G=G, t_ref=t_ref)
        # n_detect_windows walks the whole 20-yr lookback on a 10-day grid rather than scoring
        # solved apparitions, which makes it several times more expensive than everything else in
        # this function put together. It is not in the model feature list either.
        orb = add_detect_window_features(orb)
    else:
        orb = add_last_opposition_feature(orb, G=G, t_ref=t_ref)
    orb = add_detectable_apparition_features(orb, G=G, t_ref=t_ref)

    return orb


def _add_orbit_averaged_features(orb):
    """Orbit-averaged discoverability features on a 32-true-anomaly x 24-Earth-longitude grid:
    vis_orbit_mag_multi and its by-products spatial_discoverability_fraction, dec_flux_weighted,
    dec_orbit_min, frac_flux_south30, vis_orbit_flux_opp, vis_orbit_flux_multi.

    Its row chunks run on a thread pool (numpy releases the GIL); every row is independent, so
    the output is identical to the sequential loop. The 12-feature model uses none of these
    columns, so feature_engineering runs this block only under extra_features=True.
    """
    a = orb['a']
    e = np.clip(orb['e'], 0, 0.999)
    H = orb['H']
    eps_val = 1e-3

    # ============================================================================
    # ORBIT-AVERAGED FEATURES
    # ============================================================================
    
    N_ANOMALY_SAMPLES = 32
    ANOMALY_CHUNK_SIZE = 20_000   # per thread; the block is run chunk-parallel (bit-identical, rows are independent)
    N = len(orb)
    Node_rad = np.radians(orb["Node"].to_numpy(dtype=np.float64))
    Peri_rad = np.radians(orb["Peri"].to_numpy(dtype=np.float64))
    nu_arr = np.linspace(0.0, 2.0 * np.pi, N_ANOMALY_SAMPLES, endpoint=False)
    cos_nu = np.cos(nu_arr)
    obl_rad = np.radians(23.44)
    cos_obl = np.cos(obl_rad)
    sin_obl = np.sin(obl_rad)
    elong_thresh_rad = np.radians(60.0)

    # spatial_discoverability_fraction thresholds and HG phase-function constants
    # Brightness observability roll-off (replaces the old hard V <= 22.5 cut).
    # Detection efficiency tapers as objects approach the survey depth rather
    # than vanishing at one magnitude. A linear ramp (tuned via
    # modeling/parameter tuners/tune_v_lim.py) gives full weight at/brighter than
    # V_ROLLOFF_FULL and zero at/fainter than V_ROLLOFF_ZERO, i.e. weight 1 at
    # V=21.133 falling linearly to 0 at V=22.866 (width 1.733 mag).
    V_ROLLOFF_FULL = 21.1
    V_ROLLOFF_ZERO = 22.9
    # Geocentric-ecliptic-latitude observability roll-off (replaces the old hard
    # |beta| <= 25 deg cutoff). Survey coverage does not vanish abruptly at one
    # latitude; it tapers. A symmetric linear ramp (tuned via
    # modeling/parameter tuners/tune_lat_lim_rolloff.py) gives full weight at/below
    # LAT_ROLLOFF_FULL_DEG and zero at/above LAT_ROLLOFF_ZERO_DEG, i.e. weight 1 at
    # 2 deg falling linearly to 0 at 32 deg (0.5 at the 17 deg midpoint).
    LAT_ROLLOFF_FULL_DEG = 2.0
    LAT_ROLLOFF_ZERO_DEG = 32.0
    HG_A1, HG_B1 = 3.33, 0.63
    HG_A2, HG_B2 = 1.87, 1.22
    HG_G = 0.15
    PHI_FLOOR = 1e-30

    # mean_opp_dec_arr = np.empty(N, dtype=np.float64)
    spatial_disc_arr = np.empty(N, dtype=np.float64)
    dec_flux_weighted_arr = np.empty(N, dtype=np.float64)
    dec_orbit_min_arr = np.empty(N, dtype=np.float64)
    frac_flux_south30_arr = np.empty(N, dtype=np.float64)
    vis_orbit_flux_opp_arr = np.empty(N, dtype=np.float64)
    # vis_mag_timeavg_arr = np.empty(N, dtype=np.float64)
    vis_orbit_flux_multi_arr = np.empty(N, dtype=np.float64)
    vis_orbit_mag_multi_old_arr = np.empty(N, dtype=np.float64)
    vis_orbit_mag_multi_arr = np.empty(N, dtype=np.float64)

    # Earth heliocentric-longitude offsets for vis_orbit_flux_multi.
    EARTH_LON_OFFSETS_RAD = np.radians(np.array([0.0, 30.0, 60.0]))

    # --- vis_orbit_mag_multi configuration ----------------------------------
    # (NEO-aware revision; the legacy opposition-clamped version is retained as
    # vis_orbit_mag_multi_old.)
    # The legacy vis_orbit_mag_multi_old / vis_orbit_flux_multi columns clamp
    # Earth near the asteroid's longitude
    # (opposition +0/+30/+60 deg). That is correct for exterior objects but
    # actively wrong for interior / low-perihelion (q < ~1.3 AU) objects: when
    # an asteroid is sunward of Earth, "Earth at the asteroid's longitude" is
    # inferior CONJUNCTION (low solar elongation, near-"new" phase, lost in
    # glare), not opposition. Such an object's only observable window is at
    # greatest elongation, which sits at a much larger Earth-asteroid longitude
    # separation the original sampling never reaches.
    #
    # This column therefore (1) sweeps Earth around the FULL synodic circle so
    # the genuine greatest-elongation apparitions are sampled, and (2) weights every
    # geometry by an observing-efficiency ramp on solar elongation: zero below
    # ELONG_MIN_DEG (surveys do not point that near the Sun) ramping to full at
    # ELONG_FULL_DEG (encoding that little survey time is spent at low
    # elongation).
    #
    # The reported value is a *discoverability* magnitude, factored as
    #     vis_orbit_mag_multi = mag_when_observable - 2.5*log10(duty_fraction)
    # where mag_when_observable is the elongation-gated, Kepler-time-weighted
    # mean apparent V over the geometries a survey could actually point at, and
    # duty_fraction is the fraction of orbital time the object clears that gate.
    # The dilution term is the key NEO fix: an interior object's maximum possible
    # solar elongation is arcsin(Q), so deep Atens/Atiras spend little or no time
    # observable and are penalised (made fainter) accordingly -- rather than the
    # original column's spuriously faint value that came from averaging in
    # near-conjunction (back-lit, sun-glare) geometry no survey ever uses. For
    # exterior objects duty_fraction ~ the opposition half of the synodic period
    # and the penalty is small, so they behave like the legacy
    # vis_orbit_mag_multi_old.
    EARTH_LON_OFFSETS_NEOFIX_RAD = np.radians(
        np.linspace(0.0, 360.0, 24, endpoint=False)
    )
    ELONG_MIN_DEG = 60.0   # hard solar-avoidance floor (efficiency 0 below this)
    ELONG_FULL_DEG = 150.0  # full observing efficiency at/above this elongation
                            # (efficiency rises toward opposition; the model is
                            # insensitive to this endpoint over a broad range)
    DUTY_FLOOR = 1e-3      # caps the dilution penalty for never-observable orbits
                           # (~1/(n_nu*n_off) resolution -> ~+7.5 mag max penalty)

    a_np = orb['a'].to_numpy(dtype=np.float64)
    e_np = np.clip(orb['e'].to_numpy(dtype=np.float64), 0.0, 0.999)
    i_np = np.radians(orb['i'].to_numpy(dtype=np.float64))
    H_np = orb['H'].to_numpy(dtype=np.float64)
    r_t_np = a_np * (1.0 + e_np ** 2 / 2.0)

    def _orbit_chunk(start, end):
        a_c = a_np[start:end, None]
        e_c = e_np[start:end, None]
        Node_c = Node_rad[start:end, None]
        Peri_c = Peri_rad[start:end, None]
        i_c = i_np[start:end, None]
        H_c = H_np[start:end, None]
        r_t_c = r_t_np[start:end, None]

        # Heliocentric distance and Kepler-2nd-law weights at each anomaly.
        r_orb = a_c * (1.0 - e_c ** 2) / (1.0 + e_c * cos_nu[None, :])
        r_safe = np.maximum(r_orb, eps_val)
        weights = r_orb ** 2
        w_sum = np.maximum(weights.sum(axis=1), eps_val)

        # Heliocentric ecliptic coordinates of the asteroid at each nu
        u = Peri_c + nu_arr[None, :]
        cos_u = np.cos(u)
        sin_u = np.sin(u)
        cos_i = np.cos(i_c)
        sin_i_arr = np.sin(i_c)
        cos_Node = np.cos(Node_c)
        sin_Node = np.sin(Node_c)
        x_ecl = r_orb * (cos_Node * cos_u - sin_Node * sin_u * cos_i)
        y_ecl = r_orb * (sin_Node * cos_u + cos_Node * sin_u * cos_i)
        z_ecl = r_orb * sin_u * sin_i_arr

        # # mean_opp_dec
        # z_eq = y_ecl * sin_obl + z_ecl * cos_obl
        # dec_rad = np.arcsin(np.clip(z_eq / r_safe, -1.0, 1.0))
        # mean_opp_dec_arr[start:end] = np.degrees(
        #     (dec_rad * weights).sum(axis=1) / w_sum
        # )

        # spatial_discoverability_fraction
        # Place an idealised Earth at the asteroid's heliocentric ecliptic longitude
        # at r=1 AU in the ecliptic plane (z=0), then compute the apparent V via the
        # IAU HG phase function and check the geocentric ecliptic latitude.
        lambda_k = np.arctan2(y_ecl, x_ecl)
        dx = x_ecl - np.cos(lambda_k)
        dy = y_ecl - np.sin(lambda_k)
        dz = z_ecl  # Earth z = 0
        Delta_opp = np.sqrt(dx * dx + dy * dy + dz * dz)
        Delta_opp_safe = np.maximum(Delta_opp, eps_val)
        # Phase angle via law of cosines (Sun-asteroid-Earth, vertex at asteroid)
        cos_alpha = np.clip(
            (r_orb ** 2 + Delta_opp ** 2 - 1.0) / (2.0 * r_safe * Delta_opp_safe),
            -1.0, 1.0
        )
        alpha = np.arccos(cos_alpha)
        tan_half = np.tan(alpha / 2.0)
        tan_half_safe = np.maximum(tan_half, 0.0)
        phi1 = np.exp(-HG_A1 * np.power(tan_half_safe, HG_B1))
        phi2 = np.exp(-HG_A2 * np.power(tan_half_safe, HG_B2))
        phi_blend = np.maximum((1.0 - HG_G) * phi1 + HG_G * phi2, PHI_FLOOR)
        V_app = (H_c
                 + 5.0 * np.log10(r_safe * Delta_opp_safe)
                 - 2.5 * np.log10(phi_blend))
        # Geocentric ecliptic latitude (not heliocentric -- matches what surveys see)
        beta_geo = np.arcsin(np.clip(dz / Delta_opp_safe, -1.0, 1.0))
        # Linear latitude roll-off: 1 at/below LAT_ROLLOFF_FULL_DEG, tapering to
        # 0 at/above LAT_ROLLOFF_ZERO_DEG (replaces the old hard latitude mask).
        beta_deg = np.degrees(np.abs(beta_geo))
        lat_weight = np.clip(
            (LAT_ROLLOFF_ZERO_DEG - beta_deg)
            / (LAT_ROLLOFF_ZERO_DEG - LAT_ROLLOFF_FULL_DEG),
            0.0, 1.0,
        )
        # Linear brightness roll-off: 1 at/brighter than V_ROLLOFF_FULL, tapering
        # to 0 at/fainter than V_ROLLOFF_ZERO (replaces the old hard V cut).
        v_weight = np.clip(
            (V_ROLLOFF_ZERO - V_app) / (V_ROLLOFF_ZERO - V_ROLLOFF_FULL),
            0.0, 1.0,
        )
        passed = v_weight * lat_weight
        spatial_disc_arr[start:end] = (
            (passed * weights).sum(axis=1) / w_sum
        )

        # vis_orbit_flux_opp: flux-weighted mean apparent V over the orbit, in mag.
        # Averaging in linear flux (not in mag) is the physically correct way to
        # combine an exponentially distributed observable: a V=19 segment of the
        # orbit contributes ~40x more flux than a V=23 segment, so the brightest
        # portions dominate the result. The Kepler 2nd-law weights w_k = r^2 give
        # equal-time sampling.
        # Floor at 1e-30 (V ~= 75) just to keep log10 finite in pathological cases;
        # do NOT reuse the distance-scale eps_val here, which is ~6 orders of
        # magnitude larger than typical asteroid fluxes and would clamp every row.
        flux_app = np.power(10.0, -0.4 * V_app)
        flux_mean = (flux_app * weights).sum(axis=1) / w_sum
        vis_orbit_flux_opp_arr[start:end] = (
            -2.5 * np.log10(np.maximum(flux_mean, 1e-30))
        )

        # ----------------------------------------------------------------------
        # FLUX-WEIGHTED DECLINATION OF THE APPARITIONS
        # A more rigorous replacement for dec_perihelion (a single heliocentric
        # instant): asks where, in equatorial declination, the orbit's *observable*
        # light actually comes from. Uses the same idealised opposition Earth
        # (1 AU, in-ecliptic) as spatial_discoverability_fraction. The geocentric
        # vector (dx, dy, dz) is in the ecliptic frame; rotating about the x-axis
        # by the obliquity gives the equatorial z component, hence the geocentric
        # declination an observer would see at that orbital phase.
        z_eq_geo = dy * sin_obl + dz * cos_obl
        dec_geo = np.arcsin(np.clip(z_eq_geo / Delta_opp_safe, -1.0, 1.0))
        dec_geo_deg = np.degrees(dec_geo)

        # Flux x time weights. flux_app emphasises the brightest apparitions (a
        # V=19 phase outweighs a V=23 phase ~40x), while the r^2 Kepler-2nd-law
        # weights convert the uniform-in-true-anomaly samples to equal time. The
        # product is the physically correct weighting for "where does the light
        # we would actually receive over an orbit sit in declination."
        w_ft = flux_app * weights
        w_ft_sum = np.maximum(w_ft.sum(axis=1), 1e-30)

        # dec_flux_weighted: flux- and time-weighted mean geocentric declination.
        # Declination is bounded to [-90, 90] and never wraps, so the weighted
        # arithmetic mean is well defined (no circular-mean discontinuity).
        dec_flux_weighted_arr[start:end] = (
            (dec_geo_deg * w_ft).sum(axis=1) / w_ft_sum
        )

        # dec_orbit_min: southernmost declination reached at opposition anywhere on
        # the orbit (unweighted extreme) -- "how far south can this object ever get."
        dec_orbit_min_arr[start:end] = dec_geo_deg.min(axis=1)

        # frac_flux_south30: flux- and time-weighted fraction of the orbit's
        # observable light emitted while south of -30 deg declination, where major
        # northern surveys lose coverage. Distinguishes objects whose *bright*
        # apparitions fall in the deep south from those that only dip south while
        # faint near aphelion.
        south30 = (dec_geo_deg < -30.0)
        frac_flux_south30_arr[start:end] = (
            (south30 * w_ft).sum(axis=1) / w_ft_sum
        )

        # # vis_mag_timeavg: Time-weighted mean of apparent magnitude V over the orbit.
        # # As opposed to vis_orbit_flux_opp, directly averaging magnitudes prevents the 
        # # result from being overwhelmingly dominated by short-lived bright flashes 
        # # at close approaches. Time-averaging the magnitude acts as the geometric 
        # # mean of the flux, penalising objects that are very faint for the majority 
        # # of their orbits (e.g. highly eccentric NEOs).
        # mag_mean = (V_app * weights).sum(axis=1) / w_sum
        # vis_mag_timeavg_arr[start:end] = mag_mean

        # vis_orbit_flux_multi: same flux-averaging as vis_orbit_flux_opp, but averaged across
        # three Earth heliocentric-longitude positions per asteroid sample rather
        # than only at opposition. Earth at the asteroid's longitude (opposition),
        # +30 deg ahead, and +60 deg ahead.
        #
        # Physical motivation: real surveys rarely catch objects at exact opposition.
        # Sampling off-opposition geometries penalises NEOs much more than MBAs
        # because for low-r objects modest Earth offsets produce large changes in
        # geocentric distance and phase angle, while for distant objects the
        # Sun-Earth baseline is a small perturbation on the geometry.
        flux_geom_sum = flux_app  # offset = 0 (opposition); same as vis_orbit_flux_opp
        mag_geom_sum = V_app      # legacy magnitude sum (feeds vis_orbit_mag_multi_old)
        for offset_rad in EARTH_LON_OFFSETS_RAD[1:]:
            lambda_E = lambda_k + offset_rad
            dx_g = x_ecl - np.cos(lambda_E)
            dy_g = y_ecl - np.sin(lambda_E)
            # Earth z = 0, so dz_g = z_ecl (unchanged across Earth offsets)
            Delta_g = np.sqrt(dx_g * dx_g + dy_g * dy_g + z_ecl * z_ecl)
            Delta_g_safe = np.maximum(Delta_g, eps_val)
            cos_alpha_g = np.clip(
                (r_orb ** 2 + Delta_g ** 2 - 1.0) / (2.0 * r_safe * Delta_g_safe),
                -1.0, 1.0
            )
            alpha_g = np.arccos(cos_alpha_g)
            tan_half_g = np.maximum(np.tan(alpha_g / 2.0), 0.0)
            phi1_g = np.exp(-HG_A1 * np.power(tan_half_g, HG_B1))
            phi2_g = np.exp(-HG_A2 * np.power(tan_half_g, HG_B2))
            phi_blend_g = np.maximum(
                (1.0 - HG_G) * phi1_g + HG_G * phi2_g, PHI_FLOOR
            )
            V_g = (H_c
                   + 5.0 * np.log10(r_safe * Delta_g_safe)
                   - 2.5 * np.log10(phi_blend_g))
            flux_geom_sum = flux_geom_sum + np.power(10.0, -0.4 * V_g)
            mag_geom_sum = mag_geom_sum + V_g
            
        flux_geom_per_nu = flux_geom_sum / float(len(EARTH_LON_OFFSETS_RAD))
        flux_geom_mean = (flux_geom_per_nu * weights).sum(axis=1) / w_sum
        vis_orbit_flux_multi_arr[start:end] = (
            -2.5 * np.log10(np.maximum(flux_geom_mean, 1e-30))
        )
        
        mag_geom_per_nu = mag_geom_sum / float(len(EARTH_LON_OFFSETS_RAD))
        mag_geom_mean = (mag_geom_per_nu * weights).sum(axis=1) / w_sum
        vis_orbit_mag_multi_old_arr[start:end] = mag_geom_mean

        # ----------------------------------------------------------------------
        # vis_orbit_mag_multi: NEO-aware sibling of legacy vis_orbit_mag_multi_old.
        # Earth is swept around the full synodic circle (not just opposition
        # +0/+30/+60), and each (asteroid-anomaly, Earth-longitude) geometry is
        # weighted by w_kepler * obs_eff, where obs_eff ramps the solar-
        # elongation gate from 0 below ELONG_MIN_DEG to 1 at/above ELONG_FULL_DEG.
        # We accumulate the gated mean apparent V (mag_when_observable) and the
        # gate duty cycle; the final discoverability magnitude is then
        # mag_when_observable - 2.5*log10(duty_fraction). See the configuration
        # block above for the physical rationale.
        neofix_num = np.zeros(end - start, dtype=np.float64)      # sum V * w * obs_eff
        neofix_den = np.zeros(end - start, dtype=np.float64)      # sum w * obs_eff
        neofix_num_all = np.zeros(end - start, dtype=np.float64)  # fallback: ungated sum V * w
        for offset_rad in EARTH_LON_OFFSETS_NEOFIX_RAD:
            lambda_E_n = lambda_k + offset_rad
            cos_E_n = np.cos(lambda_E_n)
            sin_E_n = np.sin(lambda_E_n)
            dx_n = x_ecl - cos_E_n
            dy_n = y_ecl - sin_E_n
            # Earth z = 0, so dz_n = z_ecl (unchanged across Earth offsets)
            Delta_n = np.sqrt(dx_n * dx_n + dy_n * dy_n + z_ecl * z_ecl)
            Delta_n_safe = np.maximum(Delta_n, eps_val)
            cos_alpha_n = np.clip(
                (r_orb ** 2 + Delta_n ** 2 - 1.0) / (2.0 * r_safe * Delta_n_safe),
                -1.0, 1.0
            )
            alpha_n = np.arccos(cos_alpha_n)
            tan_half_n = np.maximum(np.tan(alpha_n / 2.0), 0.0)
            phi1_n = np.exp(-HG_A1 * np.power(tan_half_n, HG_B1))
            phi2_n = np.exp(-HG_A2 * np.power(tan_half_n, HG_B2))
            phi_blend_n = np.maximum(
                (1.0 - HG_G) * phi1_n + HG_G * phi2_n, PHI_FLOOR
            )
            V_n = (H_c
                   + 5.0 * np.log10(r_safe * Delta_n_safe)
                   - 2.5 * np.log10(phi_blend_n))

            # Solar elongation: angle Sun-Earth-asteroid as seen from Earth.
            # Earth->Sun is the unit vector -(cos lambda_E, sin lambda_E, 0);
            # Earth->asteroid is (dx_n, dy_n, z_ecl) with length Delta_n.
            cos_elong_n = np.clip(
                (-cos_E_n * dx_n - sin_E_n * dy_n) / Delta_n_safe, -1.0, 1.0
            )
            elong_deg_n = np.degrees(np.arccos(cos_elong_n))
            obs_eff = np.clip(
                (elong_deg_n - ELONG_MIN_DEG) / (ELONG_FULL_DEG - ELONG_MIN_DEG),
                0.0, 1.0
            )

            w_cell = weights * obs_eff  # (chunk, N_ANOMALY_SAMPLES)
            neofix_num += (V_n * w_cell).sum(axis=1)
            neofix_den += w_cell.sum(axis=1)
            
            neofix_num_all += (V_n * weights).sum(axis=1)

        # Base term: brightness during the observable windows (mag-space mean
        # over gated geometries). For objects with no observable geometry (deep
        # interior orbits whose max elongation never clears ELONG_MIN_DEG) fall
        # back to the ungated full-circle mean so the base stays finite; the
        # duty penalty below then drives the result faint.
        n_off = float(len(EARTH_LON_OFFSETS_NEOFIX_RAD))
        total_weight = np.maximum(w_sum * n_off, eps_val)
        mean_obs = neofix_num / np.maximum(neofix_den, eps_val)
        mean_all = neofix_num_all / total_weight
        mag_when_obs = np.where(neofix_den > eps_val, mean_obs, mean_all)

        # Duty-cycle dilution: fraction of orbital time the object clears the
        # elongation gate, converted to a magnitude penalty (duty=1 -> 0 penalty;
        # rarely/never observable -> large positive, i.e. fainter; floored so the
        # never-observable case stays finite).
        duty_fraction = neofix_den / total_weight
        duty_penalty = -2.5 * np.log10(np.maximum(duty_fraction, DUTY_FLOOR))
        vis_orbit_mag_multi_arr[start:end] = mag_when_obs + duty_penalty

    _run_row_chunks(_orbit_chunk, N, ANOMALY_CHUNK_SIZE)

    # orb["mean_opp_dec"] = mean_opp_dec_arr.astype(float)
    orb["spatial_discoverability_fraction"] = spatial_disc_arr.astype(float)
    orb["dec_flux_weighted"] = dec_flux_weighted_arr.astype(float)
    orb["dec_orbit_min"] = dec_orbit_min_arr.astype(float)
    orb["frac_flux_south30"] = frac_flux_south30_arr.astype(float)
    orb["vis_orbit_flux_opp"] = vis_orbit_flux_opp_arr.astype(float)
    # orb["vis_mag_timeavg"] = vis_mag_timeavg_arr.astype(float)
    orb["vis_orbit_flux_multi"] = vis_orbit_flux_multi_arr.astype(float)
    # orb["vis_orbit_mag_multi_old"] = vis_orbit_mag_multi_old_arr.astype(float)
    orb["vis_orbit_mag_multi"] = vis_orbit_mag_multi_arr.astype(float)

    return orb


def _add_exploratory_features(orb):
    """Adds the exploratory / superseded feature families that the current model does not use:

      * the last-three-perihelia reconstruction (vis_last_perihelion, perihelion_delta_true,
        perihelion_dec_true, vis_2nd_last_perihelion, vis_3rd_last_perihelion);
      * the 17-apparition equal-longitude solver and everything derived from it (the vis_opp_*
        family, n_valid_apparitions, opp_bright_count).

    Both depend on the orbit-averaged block, which is computed first if it is not already
    present. None of these appear in the notebook's current mlcols; they are kept because the
    demo notebook, shap_simple.py and sfs_neo.py still refer to them and because they are the
    pool a future feature search would select from, so feature_engineering(orb,
    extra_features=True) brings them all back unchanged.
    """
    if "vis_orbit_mag_multi" not in orb.columns:
        orb = _add_orbit_averaged_features(orb)
    a = orb['a']
    e = np.clip(orb['e'], 0, 0.999)
    H = orb['H']
    eps_val = 1e-3
    # Plain arrays of the elements, as the orbit-averaged block defines them (this function was
    # split out of the same monolithic routine and kept using its names).
    a_np = orb['a'].to_numpy(dtype=np.float64)
    e_np = np.clip(orb['e'].to_numpy(dtype=np.float64), 0.0, 0.999)
    i_np = np.radians(orb['i'].to_numpy(dtype=np.float64))
    H_np = orb['H'].to_numpy(dtype=np.float64)
    Node_rad = np.radians(orb["Node"].to_numpy(dtype=np.float64))
    Peri_rad = np.radians(orb["Peri"].to_numpy(dtype=np.float64))
    HG_A1, HG_B1, HG_A2, HG_B2, HG_G, PHI_FLOOR = _HG_A1, _HG_B1, _HG_A2, _HG_B2, _HG_G, _PHI_FLOOR

    # ============================================================================
    # ALIGNMENT AT LAST PERIHELION
    # ============================================================================
    # Reconstructs Sun-Earth-asteroid geometry at the most recent perihelion passage
    # prior to the catalog epoch and returns apparent V, geocentric distance, and
    # equatorial declination at that moment. Uses the catalog M and Epoch to
    # back-propagate to t_peri, an analytical Earth ephemeris (valid over the
    # decades-to-centuries lookbacks relevant here), and the IAU HG phase function.
    if "Epoch" in orb.columns and "M" in orb.columns:
        MU_SUN = 0.0002959122082855911  # AU^3 / day^2 (k^2)
        OBL_J2000 = np.radians(23.439291)
        sin_obl_p = np.sin(OBL_J2000)
        cos_obl_p = np.cos(OBL_J2000)

        M0 = np.radians(orb["M"].to_numpy(dtype=np.float64)) % (2.0 * np.pi)
        Epoch_jd = orb["Epoch"].to_numpy(dtype=np.float64)
        n_mm = np.sqrt(MU_SUN) / np.power(np.maximum(a_np, eps_val), 1.5)
        t_peri = Epoch_jd - M0 / np.maximum(n_mm, eps_val)

        # Asteroid heliocentric position at perihelion (nu = 0, r = a(1-e))
        r_peri = a_np * (1.0 - e_np)
        cos_u_p = np.cos(Peri_rad)
        sin_u_p = np.sin(Peri_rad)
        cos_i_p = np.cos(i_np)
        sin_i_p = np.sin(i_np)
        cos_Node_p = np.cos(Node_rad)
        sin_Node_p = np.sin(Node_rad)
        x_p = r_peri * (cos_Node_p * cos_u_p - sin_Node_p * sin_u_p * cos_i_p)
        y_p = r_peri * (sin_Node_p * cos_u_p + cos_Node_p * sin_u_p * cos_i_p)
        z_p = r_peri * sin_u_p * sin_i_p

        # Analytical Earth heliocentric position at t_peri (J2000 ecliptic frame).
        # The Meeus expressions below give the Sun's geocentric apparent longitude
        # Theta; Earth's heliocentric longitude is Theta + pi.
        T_jc = (t_peri - 2451545.0) / 36525.0
        M_earth = np.radians(357.52911 + 35999.05029 * T_jc)
        lambda_sun_geo = (np.radians(280.46646 + 36000.76983 * T_jc)
                          + 0.033416 * np.sin(M_earth)
                          + 0.000349 * np.sin(2.0 * M_earth))
        lambda_earth = lambda_sun_geo + np.pi
        r_earth = (1.00014061
                   - 0.01670861 * np.cos(M_earth)
                   - 0.00013957 * np.cos(2.0 * M_earth))
        x_e = r_earth * np.cos(lambda_earth)
        y_e = r_earth * np.sin(lambda_earth)
        # Earth z = 0 in the ecliptic frame

        # Geocentric vector and distance
        dx_p = x_p - x_e
        dy_p = y_p - y_e
        dz_p = z_p
        Delta_p = np.sqrt(dx_p * dx_p + dy_p * dy_p + dz_p * dz_p)
        Delta_p_safe = np.maximum(Delta_p, eps_val)
        r_peri_safe = np.maximum(r_peri, eps_val)

        # Phase angle at the asteroid vertex: cos a = (r_ast . Delta) / (|r| |Delta|)
        dot_rd = x_p * dx_p + y_p * dy_p + z_p * dz_p
        cos_alpha_p = np.clip(dot_rd / (r_peri_safe * Delta_p_safe), -1.0, 1.0)
        alpha_p = np.arccos(cos_alpha_p)
        tan_half_p = np.maximum(np.tan(alpha_p / 2.0), 0.0)
        phi1_p = np.exp(-HG_A1 * np.power(tan_half_p, HG_B1))
        phi2_p = np.exp(-HG_A2 * np.power(tan_half_p, HG_B2))
        phi_blend_p = np.maximum((1.0 - HG_G) * phi1_p + HG_G * phi2_p, PHI_FLOOR)

        V_peri = (H_np
                  + 5.0 * np.log10(r_peri_safe * Delta_p_safe)
                  - 2.5 * np.log10(phi_blend_p))

        # Equatorial declination at last perihelion
        z_eq_p = dy_p * sin_obl_p + dz_p * cos_obl_p
        dec_peri_true = np.arcsin(np.clip(z_eq_p / Delta_p_safe, -1.0, 1.0))

        orb["vis_last_perihelion"] = V_peri.astype(float)
        orb["perihelion_delta_true"] = Delta_p.astype(float)
        orb["perihelion_dec_true"] = np.degrees(dec_peri_true.astype(float))

        # --- Second-to-last perihelion -----------------------------------------
        # One orbital period (P = 2*pi / n) earlier than t_peri. To first order the
        # asteroid returns to the SAME heliocentric position at perihelion each
        # revolution (precession over a single period is negligible here), so x_p,
        # y_p, z_p are reused; only Earth has moved. Re-evaluating the apparent V at
        # this earlier passage samples a different, independent Sun-Earth-asteroid
        # geometry -- useful because whether an object was favourably placed at its
        # most recent perihelion is partly luck of the Earth phasing.
        t_peri_2 = t_peri - 2.0 * np.pi / np.maximum(n_mm, eps_val)

        T_jc2 = (t_peri_2 - 2451545.0) / 36525.0
        M_earth2 = np.radians(357.52911 + 35999.05029 * T_jc2)
        lambda_sun_geo2 = (np.radians(280.46646 + 36000.76983 * T_jc2)
                           + 0.033416 * np.sin(M_earth2)
                           + 0.000349 * np.sin(2.0 * M_earth2))
        lambda_earth2 = lambda_sun_geo2 + np.pi
        r_earth2 = (1.00014061
                    - 0.01670861 * np.cos(M_earth2)
                    - 0.00013957 * np.cos(2.0 * M_earth2))
        x_e2 = r_earth2 * np.cos(lambda_earth2)
        y_e2 = r_earth2 * np.sin(lambda_earth2)
        # Earth z = 0 in the ecliptic frame

        dx_p2 = x_p - x_e2
        dy_p2 = y_p - y_e2
        dz_p2 = z_p
        Delta_p2 = np.sqrt(dx_p2 * dx_p2 + dy_p2 * dy_p2 + dz_p2 * dz_p2)
        Delta_p2_safe = np.maximum(Delta_p2, eps_val)

        dot_rd2 = x_p * dx_p2 + y_p * dy_p2 + z_p * dz_p2
        cos_alpha_p2 = np.clip(dot_rd2 / (r_peri_safe * Delta_p2_safe), -1.0, 1.0)
        alpha_p2 = np.arccos(cos_alpha_p2)
        tan_half_p2 = np.maximum(np.tan(alpha_p2 / 2.0), 0.0)
        phi1_p2 = np.exp(-HG_A1 * np.power(tan_half_p2, HG_B1))
        phi2_p2 = np.exp(-HG_A2 * np.power(tan_half_p2, HG_B2))
        phi_blend_p2 = np.maximum((1.0 - HG_G) * phi1_p2 + HG_G * phi2_p2, PHI_FLOOR)

        V_peri2 = (H_np
                   + 5.0 * np.log10(r_peri_safe * Delta_p2_safe)
                   - 2.5 * np.log10(phi_blend_p2))

        orb["vis_2nd_last_perihelion"] = V_peri2.astype(float)

        # --- Third-to-last perihelion ------------------------------------------
        # Same construction, three orbital periods (3 * 2*pi / n) before t_peri.
        # Reuses the asteroid perihelion position (x_p, y_p, z_p); only Earth's
        # ephemeris is re-evaluated. Over three revolutions orbital precession is
        # still small for the populations here, so the fixed-perihelion-position
        # approximation continues to hold.
        t_peri_3 = t_peri - 3.0 * (2.0 * np.pi / np.maximum(n_mm, eps_val))

        T_jc3 = (t_peri_3 - 2451545.0) / 36525.0
        M_earth3 = np.radians(357.52911 + 35999.05029 * T_jc3)
        lambda_sun_geo3 = (np.radians(280.46646 + 36000.76983 * T_jc3)
                           + 0.033416 * np.sin(M_earth3)
                           + 0.000349 * np.sin(2.0 * M_earth3))
        lambda_earth3 = lambda_sun_geo3 + np.pi
        r_earth3 = (1.00014061
                    - 0.01670861 * np.cos(M_earth3)
                    - 0.00013957 * np.cos(2.0 * M_earth3))
        x_e3 = r_earth3 * np.cos(lambda_earth3)
        y_e3 = r_earth3 * np.sin(lambda_earth3)
        # Earth z = 0 in the ecliptic frame

        dx_p3 = x_p - x_e3
        dy_p3 = y_p - y_e3
        dz_p3 = z_p
        Delta_p3 = np.sqrt(dx_p3 * dx_p3 + dy_p3 * dy_p3 + dz_p3 * dz_p3)
        Delta_p3_safe = np.maximum(Delta_p3, eps_val)

        dot_rd3 = x_p * dx_p3 + y_p * dy_p3 + z_p * dz_p3
        cos_alpha_p3 = np.clip(dot_rd3 / (r_peri_safe * Delta_p3_safe), -1.0, 1.0)
        alpha_p3 = np.arccos(cos_alpha_p3)
        tan_half_p3 = np.maximum(np.tan(alpha_p3 / 2.0), 0.0)
        phi1_p3 = np.exp(-HG_A1 * np.power(tan_half_p3, HG_B1))
        phi2_p3 = np.exp(-HG_A2 * np.power(tan_half_p3, HG_B2))
        phi_blend_p3 = np.maximum((1.0 - HG_G) * phi1_p3 + HG_G * phi2_p3, PHI_FLOOR)

        V_peri3 = (H_np
                   + 5.0 * np.log10(r_peri_safe * Delta_p3_safe)
                   - 2.5 * np.log10(phi_blend_p3))

        orb["vis_3rd_last_perihelion"] = V_peri3.astype(float)

        # ====================================================================
        # BRIGHTNESS AT THE LAST 17 EQUAL-LONGITUDE APPARITIONS
        # ====================================================================
        # The solver below finds equal-heliocentric-longitude events,
        # lambda_ast == lambda_earth. For exterior objects these are true
        # oppositions, but for Earth-crossing objects they can also be inferior
        # conjunctions. We therefore treat them generically as apparitions and
        # evaluate the actual IAU (H, G) apparent V at each solved geometry.
        # Unlike perihelion this is NOT a fixed orbital position; it recurs once
        # per synodic period at times set by the Earth-asteroid longitude beat.
        # We locate the 17 most recent equal-longitude apparitions before the
        # catalog epoch and evaluate the apparent V at each.
        #
        # Method:
        #   (1) A linear mean-longitude model gives each event time to within a
        #       fraction of a synodic period. The synodic angle is
        #           psi(t) = lambda_ast(t) - lambda_earth(t),
        #       so equal-longitude events satisfy psi == 0 (mod 2*pi). The asteroid
        #       mean longitude is varpi + M(t) with varpi = Node + Peri; Earth's is
        #       the Meeus mean.
        #   (2) A few Newton steps on the TRUE psi(t) -- lambda_ast from a Kepler
        #       solve (so eccentricity and inclination enter exactly) and
        #       lambda_earth from the analytic Meeus ephemeris -- refine each time.
        #       Steps are clamped to +/- half a synodic period so each estimate stays
        #       locked to its own event window. For true oppositions V is near a
        #       local minimum, so residual timing error contributes negligibly to V.
        #
        # All 17 share the same per-object orbital elements (no precession over
        # the few-year lookback). Fully interior objects (aphelion < 1 AU) never
        # reach true opposition; for them every equal-longitude alignment is an
        # inferior conjunction and the returned V is correspondingly faint.
        TWO_PI = 2.0 * np.pi
        N_OPP = 17
        # Reliability window for the equal-longitude solver. Objects with periods
        # near 1 yr have a near-zero synodic rate, so the synodic period (the
        # spacing between successive equal-longitude apparitions) diverges and the
        # five "most recent" events can be spread over centuries. Over such spans
        # the fixed-element approximation and the low-precision analytic Earth
        # ephemeris used here are no longer trustworthy, so any solved event older
        # than this is treated as "no usable recent apparition."
        OPP_MAX_LOOKBACK_DAYS = 20.0 * 365.25

        def _wrap_pi(ang):
            return (ang + np.pi) % TWO_PI - np.pi

        def _earth_lon(t):
            # Earth heliocentric ecliptic longitude (rad): Sun's geocentric
            # longitude + pi (Meeus low-precision series).
            T = (t - 2451545.0) / 36525.0
            Me = np.radians(357.52911 + 35999.05029 * T)
            lam_sun = (np.radians(280.46646 + 36000.76983 * T)
                       + 0.033416 * np.sin(Me)
                       + 0.000349 * np.sin(2.0 * Me))
            return lam_sun + np.pi

        def _earth_xy(t):
            T = (t - 2451545.0) / 36525.0
            Me = np.radians(357.52911 + 35999.05029 * T)
            lam_sun = (np.radians(280.46646 + 36000.76983 * T)
                       + 0.033416 * np.sin(Me)
                       + 0.000349 * np.sin(2.0 * Me))
            lam_e = lam_sun + np.pi
            r_e = (1.00014061
                   - 0.01670861 * np.cos(Me)
                   - 0.00013957 * np.cos(2.0 * Me))
            return r_e * np.cos(lam_e), r_e * np.sin(lam_e)

        def _kepler_E(Marr, earr):
            # Solve M = E - e sin E (Newton). Seed handles moderate-to-high e.
            E = Marr + earr * np.sin(Marr) * (1.0 + earr * np.cos(Marr))
            for _ in range(12):
                E = E - (E - earr * np.sin(E) - Marr) / (1.0 - earr * np.cos(E))
            return E

        # Per-object columns broadcastable against the (N, 5) opposition times.
        a_col = a_np[:, None]
        e_col = e_np[:, None]
        i_col = i_np[:, None]
        H_col = H_np[:, None]
        Node_col = Node_rad[:, None]
        Peri_col = Peri_rad[:, None]
        M0_col = M0[:, None]
        Epoch_col = Epoch_jd[:, None]
        n_col = n_mm[:, None]
        cos_Node_o = np.cos(Node_col)
        sin_Node_o = np.sin(Node_col)
        cos_i_o = np.cos(i_col)
        sin_i_o = np.sin(i_col)

        # Synodic angular rate (rad/day); Earth's mean rate from the Meeus series.
        n_earth_rate = np.radians(36000.76983 / 36525.0)
        n_syn = n_mm - n_earth_rate
        n_syn_safe = np.where(np.abs(n_syn) < 1e-12, 1e-12, n_syn)
        P_syn = TWO_PI / np.abs(n_syn_safe)  # synodic period (days), > 0

        # (1) Linear initial guesses for the five most recent equal-longitude
        #     apparitions <= Epoch.
        # NOTE: this superseded solver is still anchored to the PER-OBJECT Epoch, which is the bug
        # the reference-date machinery above exists to avoid (see _reference_epoch). It is kept that
        # way deliberately so the published vis_opp_* values stay bit-identical; none of these
        # columns is in the model. Anything reusing this path should switch to _solve_apparitions.
        varpi = Node_rad + Peri_rad
        psi_E = _wrap_pi((varpi + M0) - _earth_lon(Epoch_jd))
        t_near = Epoch_jd - psi_E / n_syn_safe
        t_last = np.where(t_near > Epoch_jd, t_near - P_syn, t_near)
        k_idx = np.arange(N_OPP)
        t_opp = t_last[:, None] - k_idx[None, :] * P_syn[:, None]  # (N, 5)
        t_guess = t_opp.copy()
        half_syn = 0.5 * P_syn[:, None]

        # (2) Newton refinement on the true synodic angle, clamped to each event
        #     window.
        for _ in range(4):
            M_o = (M0_col + n_col * (t_opp - Epoch_col)) % TWO_PI
            E_o = _kepler_E(M_o, e_col)
            nu_o = 2.0 * np.arctan2(
                np.sqrt(1.0 + e_col) * np.sin(E_o / 2.0),
                np.sqrt(1.0 - e_col) * np.cos(E_o / 2.0),
            )
            u_o = nu_o + Peri_col
            cos_u_o = np.cos(u_o)
            sin_u_o = np.sin(u_o)
            x_dir = cos_Node_o * cos_u_o - sin_Node_o * sin_u_o * cos_i_o
            y_dir = sin_Node_o * cos_u_o + cos_Node_o * sin_u_o * cos_i_o
            lam_ast = np.arctan2(y_dir, x_dir)
            psi = _wrap_pi(lam_ast - _earth_lon(t_opp))
            t_opp = t_opp - psi / n_syn_safe[:, None]
            t_opp = np.clip(t_opp, t_guess - half_syn, t_guess + half_syn)

        # Final apparent V at each refined equal-longitude apparition.
        M_o = (M0_col + n_col * (t_opp - Epoch_col)) % TWO_PI
        E_o = _kepler_E(M_o, e_col)
        nu_o = 2.0 * np.arctan2(
            np.sqrt(1.0 + e_col) * np.sin(E_o / 2.0),
            np.sqrt(1.0 - e_col) * np.cos(E_o / 2.0),
        )
        r_o = a_col * (1.0 - e_col * np.cos(E_o))
        u_o = nu_o + Peri_col
        cos_u_o = np.cos(u_o)
        sin_u_o = np.sin(u_o)
        x_o = r_o * (cos_Node_o * cos_u_o - sin_Node_o * sin_u_o * cos_i_o)
        y_o = r_o * (sin_Node_o * cos_u_o + cos_Node_o * sin_u_o * cos_i_o)
        z_o = r_o * sin_u_o * sin_i_o

        x_eo, y_eo = _earth_xy(t_opp)
        dx_o = x_o - x_eo
        dy_o = y_o - y_eo
        dz_o = z_o
        Delta_o = np.sqrt(dx_o * dx_o + dy_o * dy_o + dz_o * dz_o)
        Delta_o_safe = np.maximum(Delta_o, eps_val)
        r_o_safe = np.maximum(r_o, eps_val)

        # Phase angle from the exact vectors (Sun-asteroid-Earth at the asteroid
        # vertex): cos alpha = (r_ast . (r_ast - r_earth)) / (|r_ast| |Delta|).
        dot_o = x_o * dx_o + y_o * dy_o + z_o * dz_o
        cos_alpha_o = np.clip(dot_o / (r_o_safe * Delta_o_safe), -1.0, 1.0)
        alpha_o = np.arccos(cos_alpha_o)
        tan_half_o = np.maximum(np.tan(alpha_o / 2.0), 0.0)
        phi1_o = np.exp(-HG_A1 * np.power(tan_half_o, HG_B1))
        phi2_o = np.exp(-HG_A2 * np.power(tan_half_o, HG_B2))
        phi_blend_o = np.maximum((1.0 - HG_G) * phi1_o + HG_G * phi2_o, PHI_FLOOR)

        V_opp = (H_col
                 + 5.0 * np.log10(r_o_safe * Delta_o_safe)
                 - 2.5 * np.log10(phi_blend_o))  # (N, 5), col 0 = most recent

        stale = (Epoch_col - t_opp) > OPP_MAX_LOOKBACK_DAYS

        vis_orbit = orb["vis_orbit_mag_multi"].to_numpy(dtype=np.float64)
        V_opp_clean = np.where(stale, np.nan, V_opp)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            opp_mean_clean = np.nanmean(V_opp_clean, axis=1)
            opp_mean5_clean = np.nanmean(V_opp_clean[:, :5], axis=1)
        orb["vis_opp_mean"] = opp_mean_clean.astype(float)
        orb["vis_opp_minus_orbit"] = (opp_mean_clean - vis_orbit).astype(float)
        orb["vis_opp5_minus_orbit"] = (opp_mean5_clean - vis_orbit).astype(float)
        # Recent-vs-long-run apparition brightness: is the object moving into or
        # out of its favourable phase (5-apparition mean minus 17-apparition mean).
        orb["vis_opp5_minus_opp17"] = (opp_mean5_clean - opp_mean_clean).astype(float)

        # Number of solved apparitions inside the lookback window. This is the
        # honest replacement for what the V=28 sentinel used to smuggle into
        # vis_opp_mean: roughly (lookback / synodic period), i.e. how many
        # opportunities the survey era offered at all.
        orb["n_valid_apparitions"] = (~stale).sum(axis=1).astype(float)

        # Soft count of detectable apparitions: sum over in-window events of a
        # logistic P(detect | V) centred on V = 21.5 mag with 0.5 mag width. A
        # direct physical estimate of how many apparitions a typical survey
        # could have caught; the limit and width were chosen on held-out data.
        SOFT_DET_LIMIT, SOFT_DET_WIDTH = 21.5, 0.5
        with np.errstate(over="ignore", invalid="ignore"):
            p_det = 1.0 / (1.0 + np.exp(-(SOFT_DET_LIMIT - V_opp_clean) / SOFT_DET_WIDTH))
        orb["vis_opp_soft_detect"] = np.nansum(p_det, axis=1).astype(float)

        # Trend of apparition brightness across the 17 events (mag per event,
        # positive = getting fainter toward the present), least-squares slope
        # over in-window events only.
        j_idx = np.arange(N_OPP, dtype=np.float64)[None, :].repeat(V_opp.shape[0], axis=0)
        j_idx = np.where(stale, np.nan, j_idx)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            j_mean = np.nanmean(j_idx, axis=1, keepdims=True)
            v_mean = np.nanmean(V_opp_clean, axis=1, keepdims=True)
            cov_jv = np.nansum((j_idx - j_mean) * (V_opp_clean - v_mean), axis=1)
            var_j = np.nansum((j_idx - j_mean) ** 2, axis=1)
            orb["vis_opp_slope"] = (cov_jv / np.where(var_j > 0, var_j, np.nan)).astype(float)

        # Further apparition-vs-orbit contrasts that tested as marginally helpful
        # on their own (each +0.1-0.3% held-out Poisson deviance vs the 14-feature
        # paper model on 200k rows) but were eliminated by backward selection
        # because vis_opp_minus_orbit / vis_opp_soft_detect carry the same
        # information. Kept for future experiments; see
        # modeling/tabprep_discovery/REPORT.md for the numbers.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # brightest in-window apparition minus orbit-average: single best recovery chance
            orb["vis_oppmin_minus_orbit"] = (np.nanmin(V_opp_clean, axis=1) - vis_orbit).astype(float)
            # flux-mean (rather than magnitude-mean) apparition brightness minus orbit-average;
            # weights bright apparitions more heavily. vis_opp_fluxsum is defined below, so this
            # is filled in after it is computed.
            # fraction of in-window apparitions brighter than V = 21
            n_valid_safe = np.maximum(orb["n_valid_apparitions"].to_numpy(dtype=np.float64), 1.0)
            orb["vis_opp_bright_frac"] = (np.nansum(V_opp_clean < 21.0, axis=1) / n_valid_safe).astype(float)
        # brightness at the last perihelion passage minus orbit-average (matters for high-e objects)
        orb["vis_lastperi_minus_orbit"] = (orb["vis_last_perihelion"].to_numpy(dtype=np.float64) - vis_orbit).astype(float)

        # ======================================================================
        # FEATURE SETS FOUND IN THE 2026-09 FEATURE-ENGINEERING SEARCH
        # (LightGBM, paired 5-fold CV on the full ~1.37M-row training frame;
        # gains are relative to the 14-feature set used in the paper. Notebook
        # mlcols lists are the place to enable these.)
        #
        # Paper (15 columns, two new): paper 14 minus 'e', plus
        #   'vis_opp_minus_orbit', 'perihelion_delta_true'
        #   -> Poisson deviance -1.95%, binary log-loss -0.87%
        #
        # Best 14-column set (four new, keeps vis_orbit_mag_multi):
        #   ['H','Node','a','i','vis_mid','vis_q','vis_orbit_mag_multi','vis_opp_mean',
        #    'Perihelion_direction_x_e','Perihelion_direction_y_e',
        #    'vis_opp_minus_orbit','vis_opp5_minus_opp17','vis_opp_soft_detect','perihelion_delta_true']
        #   -> Poisson deviance -2.75%, binary log-loss -0.67%
        #
        # BEST SET EVER FOUND (16 columns, unconstrained backward elimination from
        # 30 candidates; drops e, vis_opp_mean, dec_flux_weighted,
        # spatial_discoverability_fraction from the paper set):
        #   ['H','Node','a','i','vis_mid','vis_q','vis_timeavg','vis_orbit_mag_multi',
        #    'Perihelion_direction_x_e','Perihelion_direction_y_e',
        #    'vis_opp_minus_orbit','vis_opp5_minus_opp17','vis_opp_soft_detect',
        #    'perihelion_delta_true','vis_opp_slope','n_valid_apparitions']
        #   -> Poisson deviance -3.25%, binary log-loss -1.08%
        #   17 columns (+ dec_flux_weighted) was no better; 15 (- vis_opp5_minus_opp17) cost 0.45%.
        # ======================================================================

        # Count of how many of the N_OPP solved apparitions reached a "clearly
        # bright" apparent magnitude (V < VIS_OPP_BRIGHT_MAG). Linkage is a
        # best-of-N process, so the *number* of times an object was bright enough
        # to be detected is a more direct observability signal than the mean
        # brightness alone. Stale out-of-window events are NaN in V_opp_clean
        # and so never count toward this total.
        VIS_OPP_BRIGHT_MAG = 21.0
        with np.errstate(invalid="ignore"):
            orb["opp_bright_count"] = np.nansum(
                V_opp_clean < VIS_OPP_BRIGHT_MAG, axis=1
            ).astype(float)

        # ====================================================================
        # EXPERIMENTAL vis_opp VARIANTS (do not modify vis_opp_mean itself)
        # ====================================================================
        # Two independent upgrades to the recent-apparition brightness signal,
        # plus their combination:
        #
        #   (A) flux-domain averaging (vis_opp_fluxsum). vis_opp_mean is the
        #       arithmetic mean of the five apparition MAGNITUDES, which equals
        #       the GEOMETRIC mean of their fluxes -- a statistic that discards
        #       the spread across apparitions. But linkage is a best-of-N
        #       process: an object accrues an opposition if ANY apparition is
        #       bright enough, not if the typical one is. Averaging in linear
        #       flux (-2.5 log10 of the mean flux) instead is a smooth
        #       "brightest apparition" that rewards bright outliers the mean
        #       throws away (Jensen gap). It is brighter than vis_opp_mean and
        #       sits between vis_opp_mean and the brightest in-window apparition.
        #
        #   (B) observability dilution (the _disc suffix). vis_opp_mean is a
        #       pure brightness term with no notion of how OFTEN the object is
        #       well placed -- unlike vis_orbit_mag_multi, which is
        #       V_obs - 2.5 log10(f_duty). We supply the missing duty half from
        #       spatial_discoverability_fraction (the orbit-time fraction the
        #       object is simultaneously bright enough and near enough the
        #       ecliptic to be discoverable), yielding a discoverability
        #       magnitude built on the more-exact real-ephemeris brightness.
        DISC_FRAC_FLOOR = 1e-3  # caps the dilution penalty (~+7.5 mag max)
        disc_penalty = -2.5 * np.log10(
            np.maximum(
                orb["spatial_discoverability_fraction"].to_numpy(dtype=np.float64),
                DISC_FRAC_FLOOR,
            )
        )

        # (A) flux-domain mean of the in-window apparition fluxes, back in mag.
        # Floor the mean flux at 1e-30 (V ~= 75) to keep log10 finite. Stale
        # out-of-window events are NaN and drop out of the mean entirely, so an
        # object with no usable apparition yields NaN rather than a magnitude
        # built from the meaningless fixed-element reconstruction.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            flux_opp_mean = np.nanmean(np.power(10.0, -0.4 * V_opp_clean), axis=1)
            vis_opp_fluxsum = -2.5 * np.log10(np.maximum(flux_opp_mean, 1e-30))
        orb["vis_opp_fluxsum"] = vis_opp_fluxsum.astype(float)
        # flux-mean apparition brightness minus orbit-average (see contrasts block above)
        orb["vis_oppflux_minus_orbit"] = (vis_opp_fluxsum - vis_orbit).astype(float)

        # (B) discoverability-diluted variants of the two brightness terms.
        orb["vis_opp_mean_disc"] = (
            orb["vis_opp_mean"].to_numpy(dtype=np.float64) + disc_penalty
        ).astype(float)
        orb["vis_opp_fluxsum_disc"] = (vis_opp_fluxsum + disc_penalty).astype(float)
    else:
        orb["vis_last_perihelion"] = np.nan
        orb["perihelion_delta_true"] = np.nan
        orb["perihelion_dec_true"] = np.nan
        orb["vis_2nd_last_perihelion"] = np.nan
        orb["vis_3rd_last_perihelion"] = np.nan
        orb["vis_opp_mean"] = np.nan
        orb["vis_opp_minus_orbit"] = np.nan
        orb["vis_opp5_minus_orbit"] = np.nan
        orb["vis_opp5_minus_opp17"] = np.nan
        orb["n_valid_apparitions"] = np.nan
        orb["vis_opp_soft_detect"] = np.nan
        orb["vis_opp_slope"] = np.nan
        orb["vis_oppmin_minus_orbit"] = np.nan
        orb["vis_opp_bright_frac"] = np.nan
        orb["vis_lastperi_minus_orbit"] = np.nan
        orb["vis_oppflux_minus_orbit"] = np.nan
        orb["opp_bright_count"] = np.nan
        orb["vis_opp_fluxsum"] = np.nan
        orb["vis_opp_mean_disc"] = np.nan
        orb["vis_opp_fluxsum_disc"] = np.nan

    return orb


# ==============================================================================
# SURVEY-ERA APPARITION FEATURES
# ==============================================================================
# A self-contained block that solves the apparitions of the modern survey era (1998 onward) and
# derives five features. It deliberately does NOT reuse the vis_opp_* machinery above: that solver
# returns 17 apparitions with magnitudes only, while these features need 25 apparitions (28 yr at a
# typical MBA synodic period) together with each apparition's SKY POSITION, which the older code
# discards. Leaving the two paths separate keeps the published vis_opp_* columns bit-for-bit
# unchanged.
#
# Features produced:
#   gal_lat_frac_low     fraction of apparitions within 15 deg of the galactic plane
#   days_since_last_opp  days from the most recent apparition to the reference date
#   first_det_year       calendar year of the first apparition brighter than a fixed V = 21.5
#   E_full_w03           expected number of observed apparitions (the analytical detection model, 20 yr)
#   E_50yr               the same detection model over 50 yr with era limiting magnitudes back to 1976;
#                        this is the column the model uses; E_var and days_since_last_detectable_E50yr
#                        are built from the same 50-yr probabilities (E_full_w03 is reference only,
#                        days_since_last_detectable_E50yr)
#   E_var                Poisson-binomial variance of that same expectation
#   days_since_last_detectable_E50yr  days from the most recent apparition with detection probability > 0.5 to the reference date
#   year_brightest_app   calendar year of the brightest apparition in the window
#   years_since_brightest_app  the same as an elapsed duration; this is the column the model uses
#   n_detect_windows     number of separate detectable intervals in the window, from a 10-day time grid
#                        (add_detect_window_features below; modeling/feature_eng2 final round: -0.28% deviance)
#
# The last two were added in the second feature-engineering round (modeling/feature_eng2/REPORT.md):
# with orbital_period_sync they cut Poisson deviance by 1.5% on 1.17M held-out rows.
#
# Simplifications relative to the version first validated (modeling/feature_simplification/REPORT.md,
# all paired 5-fold on the full training frame, none costing performance):
#   * the never-populated 1998-2005 survey era is gone: with a 20-yr lookback every valid apparition
#     is in the survey era, so the "survey-era subset" concept is dropped (exactly identical output
#     for the current catalogue epoch);
#   * first_det_year uses a fixed V = 21.5 instead of the era-dependent limit (-0.16% deviance);
#   * the trailing-loss factor is dropped from the detection probability (-0.02% deviance);
#   * E_var uses the same magnitude width as E_full_w03 instead of its own 0.7 (-0.20% deviance).
#
# See modeling/tabprep_discovery/REPORT.md for the selection evidence and the literature behind the
# constants (Tricarico 2016 for survey depth; Denneau et al. 2013 for galactic-plane avoidance).

_N_APPARITIONS = 25
_LOOKBACK_DAYS = 20.0 * 365.25      # must match the validated configuration; do not change casually
_ERA_YEARS = (2012.0, 2020.0)       # survey-depth era boundaries ...
_ERA_VLIM = (20.5, 21.5, 22.0)      # ... and limiting magnitudes: before 2012, 2012-2019, 2020 onward
# E_50yr: the same detection model over the 50 years before the reference date. Extra apparitions solved
# (on top of the 25 shared with the other features) and the era table extended back to 1976
# (photographic surveys ~17.0, Spacewatch-era CCD ~18.5, LINEAR/NEAT ~19.5, then the table above).
# The pre-2012 entries are approximate values consistent with the reported depths of the surveys of
# each period, not figures taken from any single reference.
_N_APPARITIONS_50 = 60
_LOOKBACK_DAYS_50 = 50.0 * 365.25
_ERA50_YEARS = (1976.0, 1990.0, 1998.0, 2005.0, 2012.0, 2020.0)
_ERA50_VLIM = (17.0, 18.5, 19.5, 20.5, 21.5, 22.0)
_FIRST_DET_VLIM = 21.5      # fixed threshold for first_det_year
_MAG_WIDTH = 0.3            # sharpness of the magnitude cutoff; sharp forms beat soft ones on held-out data
# Galactic-confusion half-point and exponent. The half-point is overridable
# from the environment so that the calibration in modeling/survey_efficiency
# can be refitted against a different value of it: the measured eta_0, V_50, w
# and f_dec are all conditional on the galactic term held fixed during the fit,
# so the two cannot be varied independently.
_GAL_HALF = float(os.environ.get("ACTIVITYSCOPE_GAL_HALF", 25.0))
_GAL_POW = 2.0
_DEC_SOUTH, _DEC_NORTH, _DEC_SOFT = -30.0, 70.0, 12.0   # declination coverage taper
_GAL_LOW_EDGE = 15.0        # "in the plane" threshold, in degrees
_CONFUSION_FILES = ("bright2.pgm", os.path.join("modeling", "tabprep_discovery", "bright2.pgm"))
_RA_NGP, _DEC_NGP = np.radians(192.85948), np.radians(27.12825)
_confusion_cache = {}

# --- the reference date: a single "now" for every object ---------------------------------------
# All of these features ask a question that begins with "recently": how many apparitions in the last
# 20 (or 50) years, how long since the last one, how long has it been detectable. "Recently" has to
# be measured from somewhere, and it must be the SAME somewhere for every object, or the features
# stop being comparable across the catalogue.
#
# It is tempting to use each object's own catalogue epoch, since that is the date its elements are
# stated at -- and that is what this code originally did. But MPCORB restates almost every orbit to
# a shared standard epoch, while short-arc single-opposition objects keep an epoch near their own
# (single) observed arc. In the 2026-06 snapshot 9,879 of 1,561,163 objects carry a displaced epoch,
# and 9,876 of those 9,879 are single-opposition -- i.e. essentially the entire candidate pool this
# model exists to rank. Anchoring to the per-object epoch put their lookback window in, e.g.,
# 1982-2002: the era table scored them against pre-survey depths, days_since_last_opp came out near
# zero ("it only just had its chance"), and the model duly found nothing surprising about an object
# seen once. That is a false negative on exactly the objects we are hunting, so the window, the
# "days since" subtractions and the era clock are all anchored here instead.
#
# The per-object Epoch is still used, and must be, for its real job: propagating M0 to an arbitrary
# time. Only the meaning of "now" is centralised.
#
# Resolution order: explicit set_reference_epoch() > ACTIVITYSCOPE_REFERENCE_EPOCH_JD > the latest
# Epoch in the frame (the MPCORB standard epoch, since displaced epochs are always in the past;
# verified on the snapshot, where the maximum and the mode are the same JD 2461200.5).
_reference_epoch_override = None


def set_reference_epoch(jd):
    """Pin the date that the apparition features treat as "now" (a Julian date), or None to auto-detect.

    Auto-detection takes the latest Epoch in the frame, which is the MPCORB standard epoch for any
    full-catalogue frame. Set this explicitly when scoring a small frame that may not contain a
    standard-epoch object -- a handful of freshly designated single-opposition orbits, say -- since
    the frame's own maximum would then be an arbitrary past date.
    """
    global _reference_epoch_override
    _reference_epoch_override = None if jd is None else float(jd)


def _reference_epoch(orb):
    """The Julian date the apparition features treat as "now" for every row in `orb`."""
    if _reference_epoch_override is not None:
        return _reference_epoch_override
    env = os.environ.get("ACTIVITYSCOPE_REFERENCE_EPOCH_JD")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    ep = orb["Epoch"].to_numpy(dtype=np.float64)
    ep = ep[np.isfinite(ep)]
    if not ep.size:
        return np.nan
    t_ref = float(ep.max())
    # Safety net for the one case auto-detection cannot get right: a frame with no standard-epoch
    # row (e.g. a handful of freshly designated single-opposition orbits scored on their own), where
    # the frame's own maximum is an arbitrary past date and the window silently lands in the past.
    stale_days = (time.time() / 86400.0 + 2440587.5) - t_ref
    if stale_days > 365.0:
        warnings.warn(
            f"apparition features: reference date auto-detected as JD {t_ref:.1f}, "
            f"{stale_days / 365.25:.1f} yr in the past -- this frame appears to contain no "
            f"standard-epoch orbit. Call set_reference_epoch(jd) with the catalog retrieval date.",
            RuntimeWarning, stacklevel=2)
    return t_ref


def _load_confusion_map():
    """Gaia-DR2 'galactic confusion' map, as distributed with Find_Orb (Project Pluto).

    Provenance: star_cats/bright.c sums Gaia DR2 stellar brightness in 0.1-deg squares (one count =
    a mag-20 star); make_map.c byte-scales it with a cos(dec) normalisation; find_orb's ephem0.cpp
    consumes it as galactic_confusion(). The RA/dec -> pixel convention below (note RA runs
    BACKWARDS) reproduces that function exactly.
    """
    if "map" in _confusion_cache:
        return _confusion_cache["map"]
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in _CONFUSION_FILES:
        path = os.path.join(here, rel)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            assert fh.readline().strip() == b"P5", f"{path} is not a binary PGM"
            line = fh.readline()
            while line.startswith(b"#"):
                line = fh.readline()
            xs, ys = (int(v) for v in line.split())
            assert int(fh.readline().strip()) == 255
            img = np.frombuffer(fh.read(xs * ys), dtype=np.uint8).reshape(ys, xs).astype(np.float32)
        img = np.concatenate([img, img[:, :1]], axis=1)     # wrap column for interpolation at RA=360
        _confusion_cache["map"] = (img, xs, ys)
        return _confusion_cache["map"]
    _confusion_cache["map"] = None
    return None


def _galactic_confusion(ra_deg, dec_deg):
    """Bilinearly interpolated confusion value (0-255). Returns NaN if the map is unavailable."""
    m = _load_confusion_map()
    if m is None:
        return np.full(np.shape(ra_deg), np.nan)
    img, xs, ys = m
    ra = np.mod(np.asarray(ra_deg, dtype=np.float64), 360.0)
    dec = np.clip(np.asarray(dec_deg, dtype=np.float64), -90.0, 90.0)
    x = np.mod((720.0 - ra) * xs / 360.0 - 0.5, xs)
    y = np.clip((90.0 - dec) * ys / 180.0 - 0.5, 0.0, ys - 1.000001)
    ix = x.astype(np.int64); iy = y.astype(np.int64)
    fx = x - ix; fy = y - iy
    iy1 = np.minimum(iy + 1, ys - 1)
    return (img[iy, ix] * (1 - fx) * (1 - fy) + img[iy, ix + 1] * fx * (1 - fy)
            + img[iy1, ix] * (1 - fx) * fy + img[iy1, ix + 1] * fx * fy)


# Row-block size and worker count for the apparition solver. The solver holds well over a dozen
# (rows, n_opp) float64 temporaries at once, so on the full ~1.5M-row catalogue each one is hundreds
# of MB and every pass streams from RAM instead of cache. Splitting the catalogue into blocks keeps
# the working set resident and, because every operation in the solver is elementwise per row, gives
# bit-identical results for any block size. numpy releases the GIL inside its ufuncs, so the blocks
# also run on a thread pool for free. Set ACTIVITYSCOPE_APPARITION_WORKERS=1 to force the serial path.
#
# 2,000 rows was measured fastest (modeling/feat_eng_speed) now that feature_engineering solves all
# 60 apparitions at once rather than 25: at 2,000 x 60 each temporary is ~1 MB and a block's working
# set stays in cache, where the previous 8,000 did not. The measured curve on 60k rows / 10 cores was
# 1.81 s at 1,000, 1.28 s at 2,000, 1.33 s at 4,000, 1.83 s at 8,000 and 2.88 s at 16,000.
_APPARITION_BLOCK_ROWS = 2_000


def _run_row_chunks(fn, n_rows, chunk):
    """Call fn(start, end) for consecutive row chunks, on a thread pool when more than one core is
    available. For per-row (elementwise) work this is bit-identical to a sequential loop."""
    pairs = [(s0, min(s0 + chunk, n_rows)) for s0 in range(0, n_rows, chunk)]
    workers = _apparition_workers()
    if workers > 1 and len(pairs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda p: fn(*p), pairs))
    else:
        for s0, e0 in pairs:
            fn(s0, e0)


def _apparition_workers():
    override = os.environ.get("ACTIVITYSCOPE_APPARITION_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max(1, min(os.cpu_count() or 1, 16))


def _solve_apparitions(orb, n_opp=_N_APPARITIONS, first=0, t_ref=None):
    """Bit-exact block-parallel wrapper around _solve_apparitions_block.

    `first` is the index of the first apparition to solve (0 = the most recent), so a later call can
    extend an earlier solve without recomputing its columns: solve(n_opp=35, first=25) yields exactly
    the columns 25..59 that solve(n_opp=60) would.

    `t_ref` is the date treated as "now" (see _reference_epoch); it is resolved once here, from the
    whole frame, so that every block -- and every object -- shares one anchor.

    Splits the frame into row blocks, solves them independently (on a thread pool when more than
    one core is available), and concatenates. Every value matches what a single whole-frame call
    produces, element for element -- the solver never mixes information between rows.
    """
    if t_ref is None:
        t_ref = _reference_epoch(orb)
    n_rows = len(orb)
    workers = _apparition_workers()
    if n_rows <= _APPARITION_BLOCK_ROWS:
        return _solve_apparitions_block(orb, n_opp, first, t_ref)

    n_blocks = -(-n_rows // _APPARITION_BLOCK_ROWS)
    bounds = np.linspace(0, n_rows, n_blocks + 1).astype(np.int64)
    blocks = [orb.iloc[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]
    solve = lambda blk: _solve_apparitions_block(blk, n_opp, first, t_ref)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(solve, blocks))
    else:
        parts = [solve(blk) for blk in blocks]

    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def _slice_apparitions(G, lo, hi):
    """Apparitions lo:hi of a solved set, as _solve_apparitions would have returned them.

    The solver treats every apparition column independently -- column j is built from
    t_last - j * S_syn and nothing else -- so this is bit-identical to
    _solve_apparitions(orb, n_opp=hi - lo, first=lo). The per-row "epoch" entry is 1-D and is
    passed through unchanged.
    """
    return {k: (v[:, lo:hi] if v.ndim == 2 else v) for k, v in G.items()}


# The two apparition windows in use, the 50-yr one (_N_APPARITIONS_50) and the detectable-
# apparition one (_DET_N_APPARITIONS), happen to be the same depth today. _shared_apparition_solve
# solves the deeper of the two rather than relying on that, and each block slices what it needs.
_APPARITION_INPUT_COLUMNS = ("a", "e", "i", "Node", "Peri", "M", "Epoch", "H")


def _shared_apparition_solve(orb):
    """(G, t_ref) for the deepest apparition set any feature block needs, or (None, None) if the
    frame has no orbital elements to propagate (each block then fills its columns with NaN)."""
    if not all(c in orb.columns for c in _APPARITION_INPUT_COLUMNS):
        return None, None
    n_opp = max(_N_APPARITIONS_50, _DET_N_APPARITIONS)
    t_ref = _reference_epoch(orb)
    return _solve_apparitions(orb, n_opp=n_opp, t_ref=t_ref), t_ref


def _solve_apparitions_block(orb, n_opp=_N_APPARITIONS, first=0, t_ref=None):
    """Times and sky positions of the n_opp most recent apparitions, relative to the reference date.

    Opposition times come from a linear mean-longitude estimate refined by Newton iteration on the
    true ecliptic longitude difference. The most recent apparition is bracketed to lie at or before
    the reference date `t_ref` (see _reference_epoch -- one shared "now" for the whole catalogue,
    NOT each object's own catalogue epoch), and validity is TWO-SIDED (inside the lookback window
    and not in the future); a one-sided guard leaves roughly half of all objects with apparitions
    dated after the reference date.

    The per-object Epoch is still used for what it is for: propagating the mean anomaly to an
    arbitrary time inside ast_xyz.
    """
    if t_ref is None:
        t_ref = _reference_epoch(orb)
    f = lambda c: orb[c].to_numpy(dtype=np.float64)
    a, e = f("a"), f("e")
    inc, node, peri = np.radians(f("i")), np.radians(f("Node")), np.radians(f("Peri"))
    M0, epoch, H = np.radians(f("M")), f("Epoch"), f("H")
    n_mm = np.sqrt(0.0002959122082855911) / np.power(np.maximum(a, 1e-9), 1.5)
    n_earth = 2.0 * np.pi / 365.256

    # Everything the propagation needs that depends only on the object, not on the apparition, is
    # computed once here on the (rows,) vectors and then broadcast as a (rows, 1) column. The old
    # form tiled each element to (rows, n_opp) with B() and recomputed cos(i), sin(i), cos(Node),
    # sin(Node) and the sqrt(1 +/- e) pair from the tiled copy on every one of the five ast_xyz
    # calls. A ufunc of a tiled value is the same bits as the tiled ufunc of the value, so this
    # only removes work -- roughly a third of the solver's ufunc traffic.
    col = lambda v: v[:, None]
    a_c, e_c, peri_c, M0_c, epoch_c, n_c = (col(v) for v in (a, e, peri, M0, epoch, n_mm))
    ci, si = col(np.cos(inc)), col(np.sin(inc))
    cO, sO = col(np.cos(node)), col(np.sin(node))
    sqrt_1pe, sqrt_1me = col(np.sqrt(1.0 + e)), col(np.sqrt(1.0 - e))

    def earth_lon(t):
        """Earth's heliocentric ecliptic longitude alone -- all the refinement loop reads."""
        T = (t - 2451545.0) / 36525.0
        M = np.radians(357.52911 + 35999.05029 * T)
        return (np.radians(280.46646 + 36000.76983 * T)
                + 0.033416 * np.sin(M) + 0.000349 * np.sin(2.0 * M)) + np.pi

    def earth_xy(t):
        T = (t - 2451545.0) / 36525.0
        M = np.radians(357.52911 + 35999.05029 * T)
        lam = (np.radians(280.46646 + 36000.76983 * T)
               + 0.033416 * np.sin(M) + 0.000349 * np.sin(2.0 * M)) + np.pi
        r = 1.00014061 - 0.01670861 * np.cos(M) - 0.00013957 * np.cos(2.0 * M)
        return r * np.cos(lam), r * np.sin(lam)

    def _in_plane(t):
        """(r, cos u, sin u) at t: Kepler's equation by the validated 12 fixed Newton passes,
        then the true anomaly by the half-angle arctan2 form."""
        MM = np.mod(M0_c + n_c * (t - epoch_c) + np.pi, 2.0 * np.pi) - np.pi
        EA = MM + e_c * np.sin(MM)
        for _ in range(12):
            EA = EA - (EA - e_c * np.sin(EA) - MM) / np.maximum(1.0 - e_c * np.cos(EA), 1e-12)
        nu = 2.0 * np.arctan2(sqrt_1pe * np.sin(EA / 2), sqrt_1me * np.cos(EA / 2))
        r = a_c * (1.0 - e_c * np.cos(EA))
        u = peri_c + nu
        return r, np.cos(u), np.sin(u)

    def ast_xy(t):
        """Heliocentric ecliptic x, y at t. The refinement loop needs only the longitude, so z
        and r are not formed."""
        r, cu, su = _in_plane(t)
        return r * (cO * cu - sO * su * ci), r * (sO * cu + cO * su * ci)

    def ast_xyz(t):
        r, cu, su = _in_plane(t)
        return (r * (cO * cu - sO * su * ci), r * (sO * cu + cO * su * ci), r * su * si, r)

    rel = n_mm - n_earth
    rel = np.where(np.abs(rel) < 1e-6, np.sign(rel + 1e-12) * 1e-6, rel)
    # Mean longitude of the object AT THE REFERENCE DATE, not at its own epoch: M0 is stated at the
    # object's Epoch, so it must be advanced by n*(t_ref - Epoch) before it can be differenced
    # against Earth's longitude at t_ref. For the 99.4% of the catalogue restated to the standard
    # epoch this term is zero; for the displaced short-arc orbits it is what moves their window out
    # of the past and up to the present.
    M_ref = M0 + n_mm * (t_ref - epoch)
    lam_e0 = earth_lon(t_ref)
    d0 = np.mod(np.mod(node + peri + M_ref, 2 * np.pi) - lam_e0 + np.pi, 2 * np.pi) - np.pi
    S_syn = np.abs(2.0 * np.pi / rel)
    t_last = t_ref - d0 / rel
    t_last = t_last - np.ceil((t_last - t_ref) / S_syn) * S_syn      # force to at-or-before t_ref
    t = t_last[:, None] - np.arange(first, first + n_opp)[None, :] * S_syn[:, None]

    rel_c = col(rel)
    for _ in range(4):
        x, y = ast_xy(t)
        t = t - (np.mod(np.arctan2(y, x) - earth_lon(t) + np.pi, 2 * np.pi) - np.pi) / rel_c

    x, y, z, r = ast_xyz(t)
    xe, ye = earth_xy(t)
    dx, dy, dz = x - xe, y - ye, z
    delta = np.sqrt(dx * dx + dy * dy + dz * dz)
    obl = np.radians(23.439291)
    y_eq = dy * np.cos(obl) - dz * np.sin(obl)
    z_eq = dy * np.sin(obl) + dz * np.cos(obl)
    dec = np.arcsin(np.clip(z_eq / np.maximum(delta, 1e-12), -1, 1))
    ra = np.arctan2(y_eq, dx)
    gal_b = np.arcsin(np.clip(np.sin(dec) * np.sin(_DEC_NGP)
                              + np.cos(dec) * np.cos(_DEC_NGP) * np.cos(ra - _RA_NGP), -1, 1))
    cos_alpha = np.clip((x * dx + y * dy + z * dz) / np.maximum(r * delta, 1e-12), -1, 1)
    V = (H[:, None] + 5.0 * np.log10(np.maximum(r * delta, 1e-12))
         - 2.5 * np.log10(hg_phase(np.arccos(cos_alpha))))

    age = t_ref - t
    valid = (age <= _LOOKBACK_DAYS) & (age >= -1.0)
    # "epoch" now carries the shared reference date, broadcast per row, not the per-object Epoch.
    return dict(t=t, V=V, dec=np.degrees(dec), gal_b=np.degrees(gal_b),
                ra=np.mod(np.degrees(ra), 360.0), valid=valid,
                epoch=np.full(len(a), t_ref, dtype=np.float64))


def _era_limiting_magnitude(year):
    """Legacy hand-set limiting magnitude for the 20-yr window (fallback only)."""
    out = np.full(np.shape(year), _ERA_VLIM[0])
    for yr, vl in zip(_ERA_YEARS, _ERA_VLIM[1:]):
        out = np.where(year >= yr, vl, out)
    return out


def _era50_limiting_magnitude(year):
    """Legacy hand-set limiting magnitude for the 50-yr window (fallback only).

    NaN (no surveys) before the first era.
    """
    out = np.full(np.shape(year), np.nan)
    for yr, vl in zip(_ERA50_YEARS, _ERA50_VLIM):
        out = np.where(year >= yr, vl, out)
    return out


# --- the measured calibration -------------------------------------------------
# survey_efficiency_by_year.csv and survey_coverage_by_dec.csv are produced by
# modeling/survey_efficiency/derive_efficiency.py, which measures eta_0(t),
# V_50(t), w(t) and the declination coverage map from the MPC astrometry
# archive (see that script and modeling/survey_efficiency/REPORT.md). They
# replace both hand-set era tables and the fixed declination taper. If they are
# absent -- a checkout without the calibration, or a caller who has moved the
# working directory -- everything below falls back to the legacy constants, so
# the module keeps working, just with the old approximations.

_EFFICIENCY_FILES = ("survey_efficiency_by_year.csv",
                     os.path.join("modeling", "survey_efficiency",
                                  "survey_efficiency_by_year.csv"))
_COVERAGE_FILES = ("survey_coverage_by_dec.csv",
                   os.path.join("modeling", "survey_efficiency",
                                "survey_coverage_by_dec.csv"))
_calibration_cache = {}


def _find_calibration(paths):
    # An explicit escape hatch, for reproducing the pre-calibration behaviour
    # without deleting the CSVs (used by the paired evaluation in
    # modeling/survey_efficiency/evaluate_calibration.py).
    if os.environ.get("ACTIVITYSCOPE_DISABLE_MEASURED_EFFICIENCY"):
        return None
    # A directory holding an alternative calibration, for comparing one fit
    # against another without moving the shipped CSVs.
    alt = os.environ.get("ACTIVITYSCOPE_CALIBRATION_DIR")
    if alt:
        cand = os.path.join(alt, os.path.basename(paths[0]))
        return cand if os.path.exists(cand) else None
    here = os.path.dirname(os.path.abspath(__file__))
    for p in paths:
        for base in ("", here):
            cand = os.path.join(base, p) if base else p
            if os.path.exists(cand):
                return cand
    return None


def _efficiency_table():
    """(years, eta_0, V_50, log width) from the measured calibration, or None."""
    if "eff" not in _calibration_cache:
        path = _find_calibration(_EFFICIENCY_FILES)
        if path is None:
            _calibration_cache["eff"] = None
        else:
            df = pd.read_csv(path).sort_values("year")
            _calibration_cache["eff"] = (
                df["year"].to_numpy(dtype=np.float64),
                df["eta0"].to_numpy(dtype=np.float64),
                df["V50"].to_numpy(dtype=np.float64),
                np.log(df["width"].to_numpy(dtype=np.float64)))
    return _calibration_cache["eff"]


def _coverage_table():
    """(years, dec bin centres, factor grid) from the measured calibration, or None."""
    if "cov" not in _calibration_cache:
        path = _find_calibration(_COVERAGE_FILES)
        if path is None:
            _calibration_cache["cov"] = None
        else:
            df = pd.read_csv(path)
            df["dec_mid"] = 0.5 * (df["dec_lo"] + df["dec_hi"])
            grid = df.pivot(index="year", columns="dec_mid", values="factor").sort_index()
            _calibration_cache["cov"] = (grid.index.to_numpy(dtype=np.float64),
                                         grid.columns.to_numpy(dtype=np.float64),
                                         grid.to_numpy(dtype=np.float64))
    return _calibration_cache["cov"]


def clear_calibration_cache():
    """Forget the loaded calibration, so the next call re-reads the CSVs."""
    _calibration_cache.clear()


def _magnitude_detection(year, V, window=50):
    """The magnitude term of the detection probability, eta_0(t) * sigmoid.

    With the measured calibration this is eta_0(t) / (1 + exp((V - V_50(t))/w(t))):
    a ceiling that carries how much of the sky the surveys of that year actually
    reached, times a roll-off whose midpoint and width are measured for that
    year. Outside the calibrated span the end values are held, so a year after
    the last measured one is scored with the most recent measured behaviour
    until the calibration is rebuilt.

    Without it, the legacy behaviour: a unit ceiling and a logistic of fixed
    width _MAG_WIDTH around the hand-set era limiting magnitude, `window`
    selecting the 20-yr or 50-yr table.
    """
    tab = _efficiency_table()
    if tab is None:
        vlim = (_era50_limiting_magnitude(year) if window == 50
                else _era_limiting_magnitude(year))
        return (1.0 / (1.0 + np.exp(-np.clip((vlim - V) / _MAG_WIDTH, -30, 30))),
                np.isfinite(vlim))
    years, eta0, v50, logw = tab
    e = np.interp(year, years, eta0, left=0.0, right=eta0[-1])
    m = np.interp(year, years, v50, left=v50[0], right=v50[-1])
    w = np.exp(np.interp(year, years, logw, left=logw[0], right=logw[-1]))
    p = e / (1.0 + np.exp(np.clip((V - m) / w, -30, 30)))
    return p, np.isfinite(p)


def _threshold_magnitude(year, window=20):
    """A single limiting magnitude for the year, for the one feature that needs a hard cut.

    n_detect_windows counts runs of "detectable" samples on a time grid, which
    is a threshold question, not a probability one. With the measured
    calibration the threshold is V_50(t) -- the magnitude at which the survey
    system of that year detected half of what it could -- rather than the
    hand-set era value.
    """
    tab = _efficiency_table()
    if tab is None:
        return (_era50_limiting_magnitude(year) if window == 50
                else _era_limiting_magnitude(year))
    years, _eta0, v50, _logw = tab
    return np.interp(year, years, v50, left=v50[0], right=v50[-1])


def _declination_coverage(year, dec):
    """The declination term: the measured coverage map, else the fixed taper."""
    tab = _coverage_table()
    if tab is None:
        return np.clip(
            (1.0 / (1.0 + np.exp(-(dec - _DEC_SOUTH) / _DEC_SOFT)))
            * (1.0 / (1.0 + np.exp((dec - _DEC_NORTH) / _DEC_SOFT))), 0.0, 1.0)
    years, centres, grid = tab
    # Apparitions outside the window carry NaN declinations. The fixed taper
    # propagated those to NaN; a table lookup would index on them, so mask them
    # out here and put the NaN back at the end.
    dec = np.asarray(dec, dtype=np.float64)
    bad = ~(np.isfinite(dec) & np.isfinite(year))
    dec = np.where(bad, 0.0, dec)
    fy = np.interp(np.where(bad, years[0], year), years,
                   np.arange(len(years), dtype=np.float64))
    i0 = np.clip(np.floor(fy), 0, len(years) - 1).astype(np.intp)
    i1 = np.minimum(i0 + 1, len(years) - 1)
    wy = fy - i0
    fd = np.interp(dec, centres, np.arange(len(centres), dtype=np.float64))
    j0 = np.clip(np.floor(fd), 0, len(centres) - 1).astype(np.intp)
    j1 = np.minimum(j0 + 1, len(centres) - 1)
    wd = fd - j0
    g = ((1.0 - wy) * ((1.0 - wd) * grid[i0, j0] + wd * grid[i0, j1])
         + wy * ((1.0 - wd) * grid[i1, j0] + wd * grid[i1, j1]))
    return np.where(bad, np.nan, np.clip(g, 0.0, 1.0))


def _detection_probability(G, valid, year, window=50):
    """Per-apparition detection probability p_j = p_mag * p_gal * p_dec (zero where not valid)."""
    V = np.where(valid, G["V"], np.nan)
    conf = np.where(valid, _galactic_confusion(G["ra"], G["dec"]), np.nan)
    p_mag, mag_ok = _magnitude_detection(year, V, window=window)
    p_gal = 1.0 / (1.0 + np.power(np.maximum(conf, 0.0) / _GAL_HALF, _GAL_POW))
    dec_a = np.where(valid, G["dec"], np.nan)
    p_dec = _declination_coverage(year, dec_a)
    ok = valid & np.isfinite(p_gal) & mag_ok
    return np.where(ok, np.clip(p_mag * p_gal * p_dec, 0.0, 1.0), 0.0)


def _days_since_last_opp(G, t_ref):
    """Days from the most recent apparition inside the lookback window to the reference date,
    NaN for an object with none. Shared by add_last_opposition_feature and, for the same column,
    add_survey_era_features."""
    ok = G["valid"]
    has_any = ok.any(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        t_last = np.where(has_any, np.nanmax(np.where(ok, G["t"], -np.inf), axis=1), np.nan)
        return np.where(has_any, t_ref - t_last, np.nan).astype(float)


def add_last_opposition_feature(orb, G=None, t_ref=None):
    """Adds days_since_last_opp, the one column of the survey-era block still in the model.

    Split out so the lean feature set can have it without the superseded E_* family that
    add_survey_era_features computes around it. `G` / `t_ref` are a shared solve as in
    add_survey_era_features. Safe to call on any orbit frame.
    """
    if not all(c in orb.columns for c in _APPARITION_INPUT_COLUMNS):
        orb["days_since_last_opp"] = np.nan
        return orb
    if t_ref is None:
        t_ref = _reference_epoch(orb)
    if G is None:
        G = _solve_apparitions(orb, t_ref=t_ref)
    orb["days_since_last_opp"] = _days_since_last_opp(
        _slice_apparitions(G, 0, _N_APPARITIONS), t_ref)
    return orb


def add_survey_era_features(orb, G=None, t_ref=None):
    """Adds the survey-era apparition features. Safe to call on any orbit frame.

    `G` is an already-solved apparition set of at least _N_APPARITIONS_50 apparitions, sharing the
    reference date `t_ref` (feature_engineering solves once and passes it to every block). When it
    is None the two solves are done here, exactly as before.
    """
    needed = _APPARITION_INPUT_COLUMNS
    if not all(c in orb.columns for c in needed):
        for c in ("gal_lat_frac_low", "days_since_last_opp", "first_det_year", "years_detectable",
                  "E_full_w03", "E_var", "days_since_last_detectable_E50yr", "year_brightest_app",
                  "years_since_brightest_app", "E_50yr"):
            orb[c] = np.nan
        return orb

    # One shared "now" for every object and for both solves (the 25-apparition set and the 35 extra
    # ones E_50yr needs), so the two windows line up and no object is scored against its own past.
    if t_ref is None:
        t_ref = _reference_epoch(orb)
    if G is None:
        G25 = _solve_apparitions(orb, t_ref=t_ref)
        G2 = _solve_apparitions(orb, n_opp=_N_APPARITIONS_50 - _N_APPARITIONS,
                                first=_N_APPARITIONS, t_ref=t_ref)
    else:
        G25 = _slice_apparitions(G, 0, _N_APPARITIONS)
        G2 = _slice_apparitions(G, _N_APPARITIONS, _N_APPARITIONS_50)
    ok = G25["valid"]
    year = 2000.0 + (G25["t"] - 2451545.0) / 365.25
    n_ok = np.maximum(ok.sum(axis=1), 1)
    nan_if = lambda A: np.where(ok, A, np.nan)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        abs_b = np.abs(nan_if(G25["gal_b"]))
        orb["gal_lat_frac_low"] = (np.nansum(abs_b < _GAL_LOW_EDGE, axis=1) / n_ok).astype(float)

        has_any = ok.any(axis=1)
        orb["days_since_last_opp"] = _days_since_last_opp(G25, t_ref)

        V = nan_if(G25["V"])
        detectable = np.where(ok, V < _FIRST_DET_VLIM, False)
        any_det = detectable.sum(axis=1) > 0
        orb["first_det_year"] = np.where(
            any_det, np.nanmin(np.where(detectable, year, np.nan), axis=1), np.nan).astype(float)

        # years_detectable: the same quantity as an elapsed duration rather than a calendar year --
        # how long the object has been discoverable, counting back from the reference date. It
        # carries identical information (the two differ by a constant, the reference year) but states
        # it as a length of time, so a split on it names "has been detectable for less than N
        # years" instead of "was first detectable before calendar year Y". Replaces first_det_year
        # in the model feature list: on 1.17M rows held out from the selection subsample it is
        # -0.150% Poisson deviance (95% CI [-0.205%, -0.092%]) for +0.075% log-loss (CI
        # [-0.006%, +0.152%]). first_det_year is still computed, for the appendix and for anything
        # that references it.
        orb["years_detectable"] = (
            2000.0 + (t_ref - 2451545.0) / 365.25 - orb["first_det_year"]).astype(float)

        # --- the analytical detection model: a PRODUCT of independent factors, summed ---
        # A boosted tree splits one column at a time and cannot form products, so supplying the
        # product explicitly is what makes this worth more than its ingredients separately.
        p = _detection_probability(G25, ok, year, window=20)
        orb["E_full_w03"] = p.sum(axis=1).astype(float)   # 20-yr count, kept for reference only

        # Calendar year of the brightest apparition in the window, and the same quantity as an
        # elapsed duration. As with first_det_year/years_detectable the two differ only by the
        # constant reference year, but the duration form is stationary under retraining, so it is
        # the one in the model feature list. year_brightest_app is still computed for reference.
        i_bright = np.argmin(np.where(np.isnan(V), np.inf, V), axis=1)
        orb["year_brightest_app"] = np.where(
            has_any, year[np.arange(len(orb)), i_bright], np.nan).astype(float)
        orb["years_since_brightest_app"] = (
            2000.0 + (t_ref - 2451545.0) / 365.25 - orb["year_brightest_app"]).astype(float)

        # --- E_50yr: the same detection model over a 50-yr window ---
        # The catalogue counts every opposition ever linked, and bright objects have oppositions
        # from long before the 20-yr window; E_full_w03 cannot see them, and a SHAP analysis of
        # {E, vis_q, H} showed most of what H adds to E is exactly that pre-window history
        # (modeling/feature_eng2 on branch feature-eng-2). Extending the window to 50 yr with
        # era-appropriate limiting magnitudes puts it inside E: swapped for E_full_w03 it is
        # -0.60% Poisson deviance and -0.47% log-loss (train 200k / test 1.17M).
        # Only the 35 additional apparitions are solved; the first 25 are reused from G above
        # (the solver is elementwise per apparition, so this is identical to a single 60-solve).
        # E_var and days_since_last_detectable_E50yr use the SAME 50-yr probabilities, so the three
        # columns form one consistent construction. (The 20-yr versions were marginally better:
        # +0.12% Poisson deviance, +0.12% log-loss n.s. on 1.17M held-out rows for the pair, a
        # price accepted for a single window and era table in the paper.)
        e50 = np.zeros(len(orb)); v50 = np.zeros(len(orb)); t_det = np.full(len(orb), -np.inf)
        for Gk in (G25, G2):
            age = t_ref - Gk["t"]
            valid50 = (age <= _LOOKBACK_DAYS_50) & (age >= -1.0)
            yr = 2000.0 + (Gk["t"] - 2451545.0) / 365.25
            p50 = _detection_probability(Gk, valid50, yr, window=50)
            e50 += p50.sum(axis=1)
            v50 += (p50 * (1.0 - p50)).sum(axis=1)                       # Poisson-binomial variance
            t_det = np.maximum(t_det, np.where(p50 > 0.5, Gk["t"], -np.inf).max(axis=1))
        orb["E_50yr"] = e50.astype(float)
        orb["E_var"] = v50.astype(float)
        # Recency of the last real opportunity: days from the most recent apparition with p_j > 0.5
        # to the reference date (undefined when no apparition clears 0.5).
        orb["days_since_last_detectable_E50yr"] = np.where(np.isfinite(t_det), t_ref - t_det, np.nan).astype(float)

    return orb


# --- the compact detectable-apparition features -------------------------------
# A deliberately simple alternative to the E_50yr family: the same reconstructed
# apparitions, scored against a survey depth that is a two-constant linear ramp in
# time and a galactic-plane weight that is a linear taper in |b|. No confusion map,
# no declination map, no per-year calibration. Selected in
# modeling/compact_revision (paired 5-fold LightGBM on the full training frame).
_DET_V_NOW = 22.0            # limiting magnitude reached in _DET_Y_NOW and held thereafter
_DET_Y_NOW = 2022.0
_DET_SLOPE = 0.12            # mag per year that the limit deepened, going back in time
_DET_WINDOW_DAYS = 50.0 * 365.25
_DET_N_APPARITIONS = 60
_DET_GAL_TAPER_DEG = 30.0    # in-plane weight rises linearly from 0 at b = 0 to 1 at this |b|


def _survey_limit(year):
    """V_lim(t): the simple survey-depth ramp, 22.0 in 2022 and 0.12 mag/yr shallower before."""
    year = np.asarray(year, dtype=np.float64)
    return np.minimum(_DET_V_NOW, _DET_V_NOW - _DET_SLOPE * (_DET_Y_NOW - year))


def _galactic_taper(gal_b_deg):
    """w(b): linear taper from 0 on the galactic plane to 1 at |b| = _DET_GAL_TAPER_DEG."""
    return np.minimum(np.abs(gal_b_deg) / _DET_GAL_TAPER_DEG, 1.0)


def add_detectable_apparition_features(orb, G=None, t_ref=None):
    """Adds n_detectable, n_detectable_var, years_since_first_detectable and days_since_last_detectable.

    Every apparition j in the 50 years before the reference date gets a detection weight
        p_j = [1 + exp((V_j - V_lim(t_j)) / 0.3 mag)]^-1 * w(b_j),
    the product of a soft brightness threshold against the survey-depth ramp and the galactic
    taper. `G` is an already-solved apparition set sharing the reference date `t_ref`, as
    feature_engineering passes in; when it is None the solve is done here.
    n_detectable is the sum of p_j (an estimate of how many apparitions the object was bright
    and well placed enough to be found at), n_detectable_var the Poisson-binomial variance
    sum p_j (1 - p_j), and the two timing columns are the elapsed time since the earliest and
    since the most recent apparition with p_j > 1/2 (NaN when there is none).
    Safe to call on any orbit frame; needs the same columns as add_survey_era_features.
    """
    cols = ("n_detectable", "n_detectable_var", "years_since_first_detectable", "days_since_last_detectable")
    needed = _APPARITION_INPUT_COLUMNS
    if not all(c in orb.columns for c in needed):
        for c in cols:
            orb[c] = np.nan
        return orb
    if t_ref is None:
        t_ref = _reference_epoch(orb)
    if G is None:
        G = _solve_apparitions(orb, n_opp=_DET_N_APPARITIONS, t_ref=t_ref)
    elif G["t"].shape[1] != _DET_N_APPARITIONS:
        G = _slice_apparitions(G, 0, _DET_N_APPARITIONS)
    for c, v in zip(cols, _detectable_from_apparitions(G, t_ref)):
        orb[c] = v
    return orb


def _detectable_from_apparitions(G, t_ref, h_offset=0.0):
    """(n_detectable, n_detectable_var, years_since_first_detectable, days_since_last_detectable) from a solved apparition
    set G (as returned by _solve_apparitions with _DET_N_APPARITIONS apparitions).

    `h_offset` shifts every apparent magnitude by a constant, so the same geometry can be scored
    at a fainter H without another propagation (used by pessimistic_detectable_bounds).
    Apparitions with non-finite geometry (a degenerate clone orbit) are treated as not in the
    window.
    """
    age = t_ref - G["t"]
    inwin = (age <= _DET_WINDOW_DAYS) & (age >= -1.0) & np.isfinite(G["V"]) & np.isfinite(G["gal_b"])
    year = 2000.0 + (G["t"] - 2451545.0) / 365.25
    with np.errstate(all="ignore"):
        margin = np.where(inwin, _survey_limit(year) - (G["V"] + h_offset), np.nan)
        p_mag = 1.0 / (1.0 + np.exp(-np.clip(margin / _MAG_WIDTH, -30, 30)))
        p = np.where(inwin, p_mag * _galactic_taper(np.where(inwin, G["gal_b"], 0.0)), 0.0)
        det = p > 0.5
        any_det = det.any(axis=1)
        n_detectable = p.sum(axis=1).astype(float)
        n_var = (p * (1.0 - p)).sum(axis=1).astype(float)
        first = np.where(det, age, -np.inf).max(axis=1)
        last = np.where(det, age, np.inf).min(axis=1)
        yrs_first = np.where(any_det, first / 365.25, np.nan).astype(float)
        days_last = np.where(any_det, last, np.nan).astype(float)
    return n_detectable, n_var, yrs_first, days_last


# --- pessimistic re-scoring: orbit covariances and lower bounds on n_detectable ---
# n_detectable is a sum of steep per-apparition weights evaluated at a point estimate of an
# orbit that, for a short-arc object, is often barely constrained: a small error in a becomes
# a large error in orbital phase over the 50-year lookback, and phase decides which apparitions
# are scored against the modern survey depth. The notebook's pessimistic second pass therefore
# re-scores the strongest single-opposition candidates at the node that MINIMISES n_detectable
# over a few Gauss-Hermite nodes along the line of variation of the published MPC orbit
# covariance, each at the catalogued H and at H + 0.3 (fainter, hence less detectable).
#
# The covariances come from public.mpc_orbits on the SBN PostgreSQL mirror. The full
# covariance lives in mpc_orb_jsonb->'COM'->'covariance' as the upper triangle of a 10x10
# matrix (cov<i><j>): six cometary elements [q, e, i, node, argperi, peri_time] followed by
# four non-gravitational parameters, null unless fitted. Only the leading 6x6 block is used.
# Credentials come from the environment; nothing is stored in the repository:
#     SBN_DB_HOST        default 173.208.144.38
#     SBN_DB_PORT        default 5432
#     SBN_DB_NAME        default sbn
#     SBN_DB_USER        default pub  (read-only)
#     SBN_PUB_PASSWORD   required
_SBN_COM_NAMES = ["q", "e", "i", "node", "argperi", "peri_time"]
_GAUSS_K_DEG = 0.9856076686          # mean motion in deg/day at a = 1 AU
_MJD_TO_JD = 2400000.5
_PESSIMISTIC_H_OFFSET = 0.3          # magnitudes fainter for the pessimistic arm
_PESSIMISTIC_LOV_NODES = 5           # a short-arc covariance is ~99.9% rank one
_PESSIMISTIC_T_REF = 2461200.5       # fixed reference epoch so the bound is reproducible
_PESSIMISTIC_CHUNK_ROWS = 200_000    # node-rows per apparition solve
DETECTABLE_COLS = ("n_detectable", "n_detectable_var",
                   "years_since_first_detectable", "days_since_last_detectable")


def sbn_connect():
    """A read-only psycopg2 connection to the SBN orbit mirror (see the note above for the environment)."""
    password = os.environ.get("SBN_PUB_PASSWORD")
    if not password:
        raise RuntimeError("set SBN_PUB_PASSWORD (read-only 'pub' role) in the environment")
    import psycopg2          # not a hard dependency of the module: only the pessimistic pass needs it
    return psycopg2.connect(
        host=os.environ.get("SBN_DB_HOST", "173.208.144.38"),
        port=int(os.environ.get("SBN_DB_PORT", "5432")),
        dbname=os.environ.get("SBN_DB_NAME", "sbn"),
        user=os.environ.get("SBN_DB_USER", "pub"),
        password=password, connect_timeout=30)


def sbn_fetch_covariances(designations, conn=None):
    """{designation: dict(values, sigmas, cov, incomplete, epoch_mjd, h, stats)} from public.mpc_orbits.

    `values`, `sigmas` and the 6x6 `cov` are the cometary elements [q, e, i, node, argperi, peri_time];
    objects without a published covariance are omitted. Opens (and closes) its own connection when
    `conn` is None; pass one in when fetching in batches.
    """
    close = conn is None
    conn = conn or sbn_connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            select unpacked_primary_provisional_designation,
                   mpc_orb_jsonb->'COM',
                   mpc_orb_jsonb->'epoch_data'->>'epoch',
                   mpc_orb_jsonb->'magnitude_data'->>'H',
                   mpc_orb_jsonb->'orbit_fit_statistics'
              from public.mpc_orbits
             where unpacked_primary_provisional_designation = any(%s)""",
                    (list(designations),))
        out = {}
        for desig, com, epoch, h, stats in cur.fetchall():
            if com is None or com.get("covariance") is None:
                continue
            names = com["coefficient_names"]
            if names[:6] != _SBN_COM_NAMES:
                raise RuntimeError("unexpected element order for %s: %s" % (desig, names))
            n = len(names)
            C = np.zeros((n, n))
            bad = False
            for key, v in com["covariance"].items():
                i, j = int(key[3]), int(key[4])
                if i >= n or j >= n:
                    continue                      # the non-gravitational block
                if v is None:
                    bad = True
                    continue
                C[i, j] = C[j, i] = v
            out[desig] = dict(
                values=np.array(com["coefficient_values"][:6], dtype=float),
                sigmas=np.array(com["coefficient_uncertainties"][:6], dtype=float),
                cov=C[:6, :6], incomplete=bad,
                epoch_mjd=float(epoch), h=float(h) if h is not None else np.nan,
                stats=stats or {})
        return out
    finally:
        if close:
            conn.close()


def _com_to_keplerian(samples, epoch_mjd, h):
    """Cometary element samples (n, 6) -> the frame the apparition solver takes."""
    q, e, inc, node, argperi, tperi = samples.T
    # Far out along the line of variation a node can leave the physical region -- q below
    # zero, or e at or above one -- for an orbit whose sigma_q is a sizeable fraction of q.
    # Clamp to something still an orbit; those nodes carry little weight and non-finite
    # apparitions are dropped downstream.
    e = np.clip(e, 0.0, 0.99)
    q = np.maximum(q, 0.01)
    a = q / np.maximum(1.0 - e, 1e-6)
    n_deg = _GAUSS_K_DEG / np.power(a, 1.5)
    M = np.mod(n_deg * (epoch_mjd - tperi), 360.0)
    return pd.DataFrame(dict(a=a, e=e, i=inc, Node=node, Peri=argperi, M=M,
                             Epoch=np.full(len(a), epoch_mjd + _MJD_TO_JD),
                             H=np.full(len(a), h)))


def _lov_nodes(entry, n_nodes=_PESSIMISTIC_LOV_NODES):
    """Orbits at Gauss-Hermite nodes along the line of variation of one covariance record.

    A short-arc covariance is dominated by one direction (for one-opposition orbits the leading
    eigenvector carries a median 99.9% of the variance), so a few quadrature nodes along it
    reproduce a full 6-D Monte Carlo at a fraction of the cost. Returns (frame, weights) with
    the weights summing to one; the middle node is the nominal orbit.
    """
    C = 0.5 * (entry["cov"] + entry["cov"].T)
    w_eig, V = np.linalg.eigh(C)
    k = int(np.argmax(w_eig))
    lov = V[:, k] * np.sqrt(max(w_eig[k], 0.0))          # the 1-sigma step along the LOV
    x, wq = np.polynomial.hermite_e.hermegauss(n_nodes)  # nodes in units of sigma
    wq = wq / wq.sum()
    draws = entry["values"][None, :] + x[:, None] * lov[None, :]
    frame = _com_to_keplerian(draws, entry["epoch_mjd"], np.nan)
    frame["H"] = np.full(n_nodes, entry["h"])
    return frame, wq


def _detectable_both(frame, h_offset, t_ref=_PESSIMISTIC_T_REF):
    """The four detectable-apparition features at the frame's H and at H + h_offset, sharing one propagation."""
    set_reference_epoch(t_ref)
    G = _solve_apparitions(frame, n_opp=_DET_N_APPARITIONS, t_ref=t_ref)
    return [np.column_stack(_detectable_from_apparitions(G, t_ref, h_offset=dh))
            for dh in (0.0, h_offset)]


def pessimistic_detectable_bounds(entries, n_nodes=_PESSIMISTIC_LOV_NODES,
                                  h_offset=_PESSIMISTIC_H_OFFSET, verbose=False):
    """Lower bound on n_detectable over the orbit covariance and a fainter H, per object.

    `entries` is the {designation: record} mapping from sbn_fetch_covariances. Each object is
    scored at n_nodes Gauss-Hermite nodes along its line of variation, at the catalogued H and
    at H + h_offset (2 * n_nodes evaluations, n_nodes propagations, since H enters the apparent
    magnitude as an additive constant). Returns a DataFrame with one row per object:
    Principal_desig, n_detectable_nominal (nominal orbit, catalogued H), n_detectable_lower and
    n_detectable_upper (min and max over the scenarios), n_detectable_lower_H (the pessimistic-H
    arm alone), and the four DETECTABLE_COLS of the minimising scenario as <col>_lower, so the
    substituted row is a self-consistent orbit rather than a mix. Objects without a finite H
    are skipped; an empty DataFrame is returned when nothing can be scored.
    """
    desigs, frames = [], []
    for desig, e in entries.items():
        if not np.isfinite(e["h"]):
            continue
        f, _w = _lov_nodes(e, n_nodes)
        frames.append(f)
        desigs.append(desig)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True)
    t0 = time.time()
    F = np.empty((len(big), 2, len(DETECTABLE_COLS)))
    for lo in range(0, len(big), _PESSIMISTIC_CHUNK_ROWS):
        hi = min(lo + _PESSIMISTIC_CHUNK_ROWS, len(big))
        a, b = _detectable_both(big.iloc[lo:hi], h_offset)
        F[lo:hi, 0], F[lo:hi, 1] = a, b
        if verbose:
            print("  solved %d/%d node-rows (%.0fs)" % (hi, len(big), time.time() - t0),
                  flush=True)
    # (object, scenario, column) with the scenarios ordered [nodes at H, nodes at H + offset]
    F = F.reshape(len(desigs), n_nodes, 2, len(DETECTABLE_COLS)).transpose(0, 2, 1, 3)
    F = F.reshape(len(desigs), 2 * n_nodes, len(DETECTABLE_COLS))
    n_detectable = F[:, :, 0]
    k_min = np.argmin(np.where(np.isfinite(n_detectable), n_detectable, np.inf), axis=1)
    mid = n_nodes // 2                      # the central Gauss-Hermite node, at the catalogued H
    out = dict(Principal_desig=desigs,
               n_detectable_nominal=n_detectable[:, mid],
               n_detectable_lower=np.nanmin(n_detectable, axis=1),
               n_detectable_upper=np.nanmax(n_detectable, axis=1),
               n_detectable_lower_H=np.nanmin(n_detectable[:, n_nodes:], axis=1),
               n_evals=2 * n_nodes, n_solves=n_nodes)
    rows = np.arange(len(desigs))
    for j, c in enumerate(DETECTABLE_COLS):
        out[c + "_lower"] = F[rows, k_min, j]
    return pd.DataFrame(out)


_GRID_STEP_DAYS = 10.0          # time-grid sampling for the detectability windows
_GRID_MIN_ELONG_DEG = 60.0      # an object closer than this to the Sun is not observable, however bright
_GRID_CHUNK = 20000


def add_detect_window_features(orb):
    """Adds n_detect_windows: the number of separate intervals within the 20-yr lookback during which the
    object was brighter than the era limiting magnitude AND more than _GRID_MIN_ELONG_DEG from the Sun,
    on a _GRID_STEP_DAYS time grid.

    Unlike the apparition features, which score each solved apparition at a single instant, this walks
    the true geometry through the whole window, so it counts what was actually observable: an interior
    object's equal-longitude events next to the Sun do not count, and an apparition that stayed above the
    limit for only a few days counts the same as one that lasted months (duration itself did not add
    anything on held-out data; the count did). Same elements, Earth ephemeris and magnitude model as
    _solve_apparitions. Processed in row chunks to bound memory (~1 GB per 20k rows)."""
    needed = ("a", "e", "i", "Node", "Peri", "M", "Epoch", "H")
    if not all(c in orb.columns for c in needed):
        orb["n_detect_windows"] = np.nan
        return orb
    f = lambda c: orb[c].to_numpy(dtype=np.float64)
    a, e = f("a"), np.clip(f("e"), 0, 0.999)
    inc, node, peri = np.radians(f("i")), np.radians(f("Node")), np.radians(f("Peri"))
    M0, epoch, H = np.radians(f("M")), f("Epoch"), f("H")
    n_mm = np.sqrt(0.0002959122082855911) / np.power(np.maximum(a, 1e-9), 1.5)
    # The grid runs back from the shared reference date, not from each object's own epoch, for the
    # same reason as the apparition window (see _reference_epoch).
    t_ref = _reference_epoch(orb)
    n_t = int(round(_LOOKBACK_DAYS / _GRID_STEP_DAYS)) + 1
    offsets = np.arange(n_t)[None, :] * _GRID_STEP_DAYS
    out = np.full(len(orb), np.nan)

    def earth_xy(t):
        T = (t - 2451545.0) / 36525.0
        M = np.radians(357.52911 + 35999.05029 * T)
        lam = (np.radians(280.46646 + 36000.76983 * T)
               + 0.033416 * np.sin(M) + 0.000349 * np.sin(2.0 * M)) + np.pi
        r = 1.00014061 - 0.01670861 * np.cos(M) - 0.00013957 * np.cos(2.0 * M)
        return r * np.cos(lam), r * np.sin(lam)

    for s0 in range(0, len(orb), _GRID_CHUNK):
        sl = slice(s0, min(len(orb), s0 + _GRID_CHUNK))
        B = lambda v: v[sl][:, None]
        t = t_ref - offsets
        MM = np.mod(B(M0) + B(n_mm) * (t - B(epoch)) + np.pi, 2.0 * np.pi) - np.pi
        E_ = B(e)
        EA = MM + E_ * np.sin(MM)
        for _ in range(10):
            EA = EA - (EA - E_ * np.sin(EA) - MM) / np.maximum(1.0 - E_ * np.cos(EA), 1e-12)
        nu = 2.0 * np.arctan2(np.sqrt(1 + E_) * np.sin(EA / 2), np.sqrt(1 - E_) * np.cos(EA / 2))
        r = B(a) * (1.0 - E_ * np.cos(EA))
        u = B(peri) + nu
        cu, su, ci, si, cO, sO = np.cos(u), np.sin(u), np.cos(B(inc)), np.sin(B(inc)), np.cos(B(node)), np.sin(B(node))
        x, y, z = r * (cO * cu - sO * su * ci), r * (sO * cu + cO * su * ci), r * su * si
        xe, ye = earth_xy(t)
        dx, dy, dz = x - xe, y - ye, z
        delta = np.sqrt(dx * dx + dy * dy + dz * dz)
        with np.errstate(all="ignore"):
            cos_alpha = np.clip((x * dx + y * dy + z * dz) / np.maximum(r * delta, 1e-12), -1.0, 1.0)
            V = B(H) + 5.0 * np.log10(np.maximum(r * delta, 1e-12)) - 2.5 * np.log10(hg_phase(np.arccos(cos_alpha)))
            re_ = np.sqrt(xe * xe + ye * ye)
            cos_elong = np.clip(-(xe * dx + ye * dy) / np.maximum(re_ * delta, 1e-12), -1.0, 1.0)
            year = 2000.0 + (t - 2451545.0) / 365.25
            det = (V < _threshold_magnitude(year)) & (cos_elong < np.cos(np.radians(_GRID_MIN_ELONG_DEG)))
        d = det.astype(np.int8)
        out[sl] = d[:, 0] + ((d[:, 1:] == 1) & (d[:, :-1] == 0)).sum(axis=1)   # number of runs of consecutive detectable samples
    orb["n_detect_windows"] = out.astype(float)
    return orb


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def load_all_databases():
    """
    Load all orbit databases (MPC, AstDyS, JPL) and merge them.
    
    Returns
    -------
    pd.DataFrame
        Combined orbit dataframe with all databases merged
    """
    print("Loading MPC orbits...")
    orb = load_mpc_orbits()
    
    print("Loading astrometry counts...")
    orb = load_astrometry_counts(orb)
    print("Applying nights overrides...")
    orb = apply_nights_overrides(orb)
    
    print("Loading AstDyS orbits...")
    astdys = load_astdys_orbits()
    
    print("Loading JPL orbits...")
    jpl = load_jpl_orbits()
    
    print("Comparing with AstDyS...")
    orb = compare_with_astdys(orb, astdys)
    
    print("Comparing with JPL...")
    orb = compare_with_jpl(orb, jpl)
    
    print("Computing database differences...")
    orb = compute_database_differences(orb)
    
    print("Applying number of oppositions overrides...")
    orb = apply_num_opps_overrides(orb)
    
    # Apply robust H: take the median when all three databases provide a value (to drop outliers),
    # else fall back to the dimmest (max) across available databases.
    # This must happen here, before feature_engineering is called, so that all
    # visibility features (vis_typ, vis_q, etc.) are computed using the combined H.
    # At this point orb["H"] is still the raw MPC value (== H_MPC); manual photometry
    # fixes are applied just below, after this cross-database combination.
    print("Applying robust H (median if 3 values, else max (dimmest) across MPC/AstDyS/JPL; corrections applied next)...")
    h_cols = orb[["H", "H_astdys", "H_jpl"]]
    has_3 = h_cols.notna().sum(axis=1) == 3
    orb["H"] = np.where(has_3, h_cols.median(axis=1), h_cols.max(axis=1))

    # Not sure which database is doing it, but drop any rows with H values == -9.99
    orb = orb[orb["H"] != -9.99]

    print("Applying magnitude corrections...")
    orb = apply_magnitude_corrections(orb)
    
    print("Adding training targets...")
    orb = add_training_targets(orb)
    
    return orb


# ==============================================================================
# EXTENSION DIFFICULTY CLASSIFIER
# ==============================================================================

def train_extension_difficulty_classifier(orb_pred, final, orb, filter_csv="filter out unless updated.csv"):
    """
    Train extension difficulty classifier using iterative refinement.
    
    Extension difficulty quantifies objects that are challenging to extend due to:
    - Too uncertain to recover with ITF (Isolated Tracklet File) techniques
    - Too much of a "stretch" linkage that may represent a chimera orbit 
      (two different objects incorrectly linked together)
    
    The classifier uses the principle that highly-rated objects from prior ML 
    models that remain single-opposition are more representative of "difficult 
    to extend" objects. The difficulty is based primarily on astrometry metadata 
    (e.g., number of nights, arc length) rather than orbital elements.
    
    Parameters
    ----------
    orb_pred : pd.DataFrame
        Orbit predictions dataframe with prob, Num_opps, and astrometry metadata
    final : pd.DataFrame
        Final results dataframe to apply predictions to
    filter_csv : str, optional
        Path to filter CSV file (default: "filter out unless updated.csv")
    
    Returns
    -------
    tuple
        (orb_pred, final)
        - orb_pred: Updated orb_pred with extension_difficulty column
        - final: Updated final with extension_difficulty column
    """
    for df in (orb_pred, final):
        df["v_mag_gap"] = df["v_mag_max"] - df["v_mag_min"]
        df["second_minmax_gap"] = df["v_mag_second_max"] - df["v_mag_second_min"]
        df["v_mag_gap_1"] = df["v_mag_max"] - df["v_mag_avg"]
        df["v_mag_gap_2"] = df["v_mag_avg"] - df["v_mag_min"]

    def _weighted(df, target_n=None, frac=None, default_weight = 1.0):
        """Attach a per-row sampling weight so the whole list is used instead of
        being subsampled, then fed to XGBoost via sample_weight.

        - target_n: weight = target_n / len(df), reproducing the effective
          contribution of df.sample(target_n) without dropping any rows.
        - frac: weight = frac, reproducing df.sample(frac=...).
        - neither: weight = 1.0, for lists that were already used in full.
        """
        n = len(df)
        if frac is not None:
            w = frac
        elif target_n is not None and n > 0:
            w = target_n / n
        else:
            w = default_weight
        return df.assign(weight=w)

    # Load filter list, only keep the mislinkage comments
    filter_out_unless_updated = pd.read_csv(filter_csv)
    filter_out_unless_updated = filter_out_unless_updated[filter_out_unless_updated["reason"].str.contains("misl", case=False, na=False)]
    multiopp_mislinkages = pd.read_csv("filter until further notice.csv")
    multiopp_mislinkages = multiopp_mislinkages[multiopp_mislinkages["reason"].str.contains("misl", case=False, na=False)]
    
    # Exclude recent observations and objects without astrometry metadata
    # Anything with E2026 may be too recent to have had the ITF community complete the extension if it is possible to extend.
    
    use_to_train_misl = orb_pred[
        (orb_pred["Ref"].str[0:5] != "E2026") & 
        (orb_pred["nights_total"].notna())
    ]
    
    # =========================================================================
    # POSITIVE CLASS: High extension difficulty objects
    # (too uncertain for ITF recovery or potential chimera orbits)
    # =========================================================================
    
    # Single opposition, high probability objects, we will include all with fewer than 4 nights and a sample of those with 4 or more nights
    poss_misl_or_unc_1opp = use_to_train_misl[
        (use_to_train_misl["prob"] > 0.975) & 
        (use_to_train_misl["Num_opps"] == 1)
    ]
    poss_misl_or_unc_lt4nights = poss_misl_or_unc_1opp[
        poss_misl_or_unc_1opp["nights_total"] < 4
    ]
    poss_misl_or_unc_ge4nights = poss_misl_or_unc_1opp[
        poss_misl_or_unc_1opp["nights_total"] >= 4
    ]
    # but drop from this 4+ nights group any with a pre-existing extension_difficulty lower than 0.01 (likely okay)
    poss_misl_or_unc_ge4nights = poss_misl_or_unc_ge4nights[
        poss_misl_or_unc_ge4nights["extension_difficulty"] >= 0.01
    ]

    # Objects with U = 9
    poss_unc_U = poss_misl_or_unc_1opp[poss_misl_or_unc_1opp["U"] == 9]

    # high mag residuals likely
    poss_misl_or_unc_1opp_high_magresids = use_to_train_misl[
        (use_to_train_misl["v_mag_gap"] > 3)
        & (use_to_train_misl["v_mag_gap_1"] > 0.5)
        & (use_to_train_misl["v_mag_gap_2"] > 1.5)
        & (use_to_train_misl["second_minmax_gap"] > 1.5)
        & (use_to_train_misl["Arc_length"].between(7, 30))
        & (use_to_train_misl["nights_total"] <= 4)
        & (use_to_train_misl["Perihelion_dist"].between(1.6, 3.5))]
    
    # Objects in filter list and 2-3 opposition high-prob objects.
    poss_misl_or_unc_named = orb.merge(
        filter_out_unless_updated.rename(columns={"Object": "Principal_desig"})[["Principal_desig", "Arc_length"]],
        on=["Principal_desig", "Arc_length"],
        how="inner"
    )
    poss_multi_opp_mislinkages_named = orb.merge(
        multiopp_mislinkages.rename(columns={"Object": "Principal_desig"})[["Principal_desig"]],
        on=["Principal_desig"],
        how="inner"
    )
    print(f"Positive examples from filter list: {len(poss_misl_or_unc_named)}")
    print(f"Positive examples from multi-opp mislinkages: {len(poss_multi_opp_mislinkages_named)}")
    
    poss_misl_or_unc_23opp = use_to_train_misl[
        (use_to_train_misl["prob"] > 0.985) & 
        (use_to_train_misl["Num_opps"].between(2, 3)) &
        (use_to_train_misl["nights_total"] <= 10)
    ]

    # all objects with longest arc less than 5 days yet either 2 or 3 opps and nights_total <=7
    poss_misl_or_unc_lt5day_arc_23opp = use_to_train_misl[
        (use_to_train_misl["prob"] > 0.98)
        & (use_to_train_misl["Num_opps"].between(2, 3))
        & (use_to_train_misl["longest_opp_arc"] < 5)
        & (use_to_train_misl["nights_total"] <= 7)
    ]
    
    # Combine positive examples
    poss_misl_or_unc = pd.concat([
        _weighted(poss_misl_or_unc_23opp),
        _weighted(poss_misl_or_unc_lt4nights),
        _weighted(poss_misl_or_unc_ge4nights, frac=0.2),
        _weighted(poss_misl_or_unc_named),
        _weighted(poss_multi_opp_mislinkages_named),
        _weighted(poss_misl_or_unc_1opp_high_magresids),
        _weighted(poss_misl_or_unc_lt5day_arc_23opp, default_weight=5),
        _weighted(poss_unc_U, frac=0.4)
    ])

    # Remove duplicates, keeping the highest-weight copy of each object
    poss_misl_or_unc = poss_misl_or_unc.sort_values(
        "weight", ascending=False, kind="stable"
    )
    poss_misl_or_unc = poss_misl_or_unc[
        ~poss_misl_or_unc.index.duplicated(keep='first')
    ]
    
    # Filter out likely recoverable objects
    # 5+ nights over 12+ days in single opp is almost certainly not high difficulty
    poss_misl_or_unc = poss_misl_or_unc[~(
        (poss_misl_or_unc["Arc_length"] >= 12) & 
        (poss_misl_or_unc["nights_total"] >= 5) & 
        (poss_misl_or_unc["Num_opps"] == 1)
    )]
    
    # Objects with second opposition having 2+ nights have low extension difficulty
    poss_misl_or_unc = poss_misl_or_unc[
        ~(poss_misl_or_unc["opp_with_second_most_nights"] > 1)
    ]
    
    poss_misl_or_unc["label"] = 1
    print(f"Positive examples (high extension difficulty): {len(poss_misl_or_unc)}")
    
    # =========================================================================
    # NEGATIVE CLASS: Low extension difficulty (likely recoverable/reliable)
    # =========================================================================
    
    likely_okay = use_to_train_misl[
        (use_to_train_misl["prob"] < 0.6) & 
        (use_to_train_misl["Num_opps"] < 3)
    ]
    likely_okay_heavier = likely_okay[
        (likely_okay["nights_total"] >= 4) | 
        ((likely_okay["nights_total"] == 3) & (likely_okay["Arc_length"].between(11,22)))
    ]
    likely_okay_gt3_opps = use_to_train_misl[use_to_train_misl["Num_opps"] >= 3]
    likely_okay_4night_withU = likely_okay[
        (likely_okay["nights_total"] == 4)
        &(likely_okay["U"]<9)
        ]
    likely_okay_gt5_nights_single_opp = likely_okay[
        (likely_okay["nights_total"] >= 5) & 
        (likely_okay["Num_opps"] == 1)
    ]
    likely_okay_gt4_opps_short_init_arc = use_to_train_misl[
        (use_to_train_misl["Num_opps"] >= 4) & 
        (use_to_train_misl["longest_opp_arc"] < 6)
    ]
    likely_okay_3opp = use_to_train_misl[
        (use_to_train_misl["prob"] < 0.97) & 
        (use_to_train_misl["Num_opps"] == 3)
    ]

    # objects with opp_with_second_most_nights > 1
    likely_okay_2_nights_second_opp = use_to_train_misl[
        use_to_train_misl["opp_with_second_most_nights"] == 2
    ]

    # low mag and astrometric residuals
    likely_okay_lowresids = likely_okay[
        (likely_okay["second_minmax_gap"] < 0.8)
      & (likely_okay["Arc_length"].between(11, 17))
      & (likely_okay["nights_total"].between(3,5))
      & (likely_okay["prob"]<0.4)
      & (likely_okay["rms"] < 0.09)]
    
    # Balanced weighting of negative examples (full lists, weighted in place of
    # the prior per-list subsampling)
    likely_okay = pd.concat([
        _weighted(likely_okay_gt5_nights_single_opp, 5000),
        _weighted(likely_okay, 5000),
        _weighted(likely_okay_heavier, 7000),
        _weighted(likely_okay_gt3_opps, 2000),
        _weighted(likely_okay_gt4_opps_short_init_arc, 500),
        _weighted(likely_okay_4night_withU, 8000),
        _weighted(likely_okay_3opp, 500),
        _weighted(likely_okay_lowresids),
        _weighted(likely_okay_2_nights_second_opp, 2000),
    ])

    # Remove duplicates, keeping the highest-weight copy of each object
    likely_okay = likely_okay.sort_values(
        "weight", ascending=False, kind="stable"
    )
    likely_okay = likely_okay[
        ~likely_okay.index.duplicated(keep='first')
    ]

    likely_okay["label"] = 0
    print(f"Negative examples (low extension difficulty): {len(likely_okay)}")

    # =========================================================================
    # CLASSIFIER TRAINING WITH ITERATIVE REFINEMENT
    # =========================================================================
    
    misl_training = pd.concat([poss_misl_or_unc, likely_okay])

    # Define feature columns
    misl_cols = [
        "U", "longest_opp_arc", "longest_gap_arc", "second_longest_gap_arc", 
        "shortest_gap_arc", "opposition_count", "opp_with_most_nights", 
        "opp_with_second_most_nights", "other_opps", "nights_total", 
        "Num_opps", "Num_obs", "Arc_length", #"v_mag_gap", "second_minmax_gap",
        
        "label"
    ]
    misl_cols_simple = ["U", "Num_opps", "Num_obs", "Arc_length"]
    
    # Initialize classifiers
    xgb_misl = XGBClassifier()
    xgb_misl_simple = XGBClassifier()
    
    def train_and_predict():
        """Helper function to train both classifiers and make predictions."""
        misl_training_mlcols = misl_training[misl_cols]
        
        # Train main classifier
        xgb_misl.fit(
            misl_training_mlcols.drop(columns=["label"]),
            misl_training_mlcols["label"],
            sample_weight=misl_training["weight"],
        )
        final["extension_difficulty"] = xgb_misl.predict_proba(
            final[misl_cols[:-1]].astype(float)
        )[:, 1]
        
        # Train simplified classifier (for missing astrometry data)
        xgb_misl_simple.fit(
            misl_training_mlcols[misl_cols_simple],
            misl_training_mlcols["label"],
            sample_weight=misl_training["weight"],
        )
        final["extension_difficulty_simple"] = xgb_misl_simple.predict_proba(
            final[misl_cols_simple].astype(float)
        )[:, 1]
        
        # Use simple classifier predictions where astrometry data is missing
        final.loc[final["nights_total"].isna(), "extension_difficulty"] = \
            final.loc[final["nights_total"].isna(), "extension_difficulty_simple"]
        final.drop(columns=["extension_difficulty_simple"], inplace=True)
    
    # Initial training
    train_and_predict()
    
    # Iterative refinement: Remove likely mislabeled examples
    misl_training["mr_temp"] = xgb_misl.predict_proba(
        misl_training[misl_cols[:-1]].astype(float)
    )[:, 1]

    # First refinement: Remove obvious mislabels
    misl_training = misl_training[~(
        (misl_training["label"] == 1) & (misl_training["mr_temp"] < 0.15)
    )]
    misl_training = misl_training[~(
        (misl_training["label"] == 0) & (misl_training["mr_temp"] > 0.88)
    )]
    
    # Retrain after first refinement
    train_and_predict()

    # Iterative refinement: Remove likely mislabeled examples
    misl_training["mr_temp"] = xgb_misl.predict_proba(
        misl_training[misl_cols[:-1]].astype(float)
    )[:, 1]

    # Second refinement: More aggressive filtering of positive class
    misl_training = misl_training[~(
        (misl_training["label"] == 1) & (misl_training["mr_temp"] < 0.04)
    )]
    print(f"Training set after refinement: {len(misl_training)}")
    
    # Final training
    train_and_predict()
    
    # Apply predictions to orb_pred
    orb_pred["extension_difficulty"] = xgb_misl.predict_proba(
        orb_pred[misl_cols[:-1]].astype(float)
    )[:, 1]

    # make sure that both orb_pre and final have extension_difficulty rounded to nearest 0.000001 since we think this is more readable than sci notation
    orb_pred["extension_difficulty"] = orb_pred["extension_difficulty"].round(6)
    final["extension_difficulty"] = final["extension_difficulty"].round(6)
    
    return orb_pred, final

def calc_jd(year, month, day):
    import numpy as np
    y = year.copy()
    m = month.copy()
    mask = m <= 2
    y[mask] -= 1
    m[mask] += 12
    A = np.floor(y / 100)
    B = 2 - A + np.floor(A / 4)
    B[y < 1582] = 0
    B[(y == 1582) & (month < 10)] = 0
    B[(y == 1582) & (month == 10) & (day <= 4)] = 0
    return np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1)) + day + B - 1524.5

def getFinal(orb_pred,known_strongly_suspected_active_objects):
    final = orb_pred.copy()

    # renames of Perihelion_dist to q and Aphelion_dist to Q for brevity
    final.rename(columns={"Perihelion_dist":"q","Aphelion_dist":"Q"},inplace=True)
    final.set_index("Principal_desig", inplace=True)

    # Calculate "quantile deficit"
    final['DeltaQ'] = final["quantile_Opps"]-final["Num_opps"]
    final.sort_values("DeltaQ",ascending=False, inplace=True)

    # Mark which ones are known or strongly suspected
    final["Known / Strong Suspect"] = final.index.isin(known_strongly_suspected_active_objects)
    return final
