import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from astropy import units as u
from astropy.constants import c
from dust_extinction.parameter_averages import F99
from pathlib import Path


# Constants
C_AA_S = 2.99792458e18
FNU0_AB = 3631.0e-23

# Reference-band flux using Gopreaux's current zero point
zp_peak_gopreaux = SN08D.zps[SN08D.info["peak_filt"]]

f_peak_ref_gopreaux = (
    zp_peak_gopreaux
    * 1e-11
    * 10**(-0.4 * SN08D.info["peak_mag"])
)

f_peak_ref_gopreaux_log = np.log10(f_peak_ref_gopreaux)

# Phase grid
phase_min, phase_max, phase_step = -19, 49, 2
phases = np.arange(
    phase_min,
    phase_max + phase_step,
    phase_step
)

# Corrected results
F_bol = np.zeros(len(phases))
F_up = np.zeros(len(phases))
F_low = np.zeros(len(phases))

F_UV = np.zeros(len(phases))
F_OPT = np.zeros(len(phases))
F_NIR = np.zeros(len(phases))

# Keep the original result for comparison
F_bol_gopreaux = np.zeros(len(phases))

uv_opt_boundary = 0.5 * (WLE["U"] + WLE["B"])
opt_nir_boundary = 0.5 * (WLE["i"] + WLE["J"])

for i, phase in enumerate(phases):

    x, shifted_flux, sigma = snmodel.predict_sed(
        wavelength_min=2220,
        wavelength_max=22114,
        phase=phase,
        show=False,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(shifted_flux)
        & np.isfinite(sigma)
    )

    x = x[valid]
    shifted_flux = shifted_flux[valid]
    sigma = sigma[valid]

    # Flux produced using the current gopreaux convention
    f_lambda_gopreaux = 10**(
        shifted_flux + f_peak_ref_gopreaux_log
    )

    f_lambda_gopreaux_upper = 10**(
        shifted_flux
        + f_peak_ref_gopreaux_log
        + sigma
    )

    f_lambda_gopreaux_lower = 10**(
        shifted_flux
        + f_peak_ref_gopreaux_log
        - sigma
    )

    # Current Gopreaux zero point
    f0_gopreaux = 1e-23 * 3e18 / x

    # Correct AB f_lambda zero point
    f0_ab = FNU0_AB * C_AA_S / x**2

    # Wavelength-dependent correction
    correction = f0_ab / f0_gopreaux

    # Corrected physical flux
    f_lambda = f_lambda_gopreaux * correction
    f_lambda_upper = (
        f_lambda_gopreaux_upper * correction
    )
    f_lambda_lower = (
        f_lambda_gopreaux_lower * correction
    )

    # Original gopreaux bolometric flux
    F_bol_gopreaux[i] = np.trapezoid(
        f_lambda_gopreaux,
        x
    )

    # Corrected bolometric flux
    F_bol[i] = np.trapezoid(
        f_lambda,
        x
    )

    F_up[i] = np.trapezoid(
        f_lambda_upper,
        x
    )

    F_low[i] = np.trapezoid(
        f_lambda_lower,
        x
    )

    uv = x <= uv_opt_boundary
    optical = (
        (x > uv_opt_boundary)
        & (x < opt_nir_boundary)
    )
    nir = x >= opt_nir_boundary

    F_UV[i] = np.trapezoid(
        f_lambda[uv],
        x[uv]
    )

    F_OPT[i] = np.trapezoid(
        f_lambda[optical],
        x[optical]
    )

    F_NIR[i] = np.trapezoid(
        f_lambda[nir],
        x[nir]
    )




DISTANCE_MPC = 31.0
EBV_HOST = 0.60
RV = 3.10
f99 = F99(Rv=RV)

# UVW1/U/B/V are UVOT/Johnson-like bands.
# r'/i' are the FLWO Sloan-like bands.
# J/H/Ks are PAIRITEL/2MASS bands.

LAMBDA_EFF = {
    "UVW2": 1928.0,
    "UVM2": 2246.0,
    "UVW1": 2600.0,
    "U":    3465.0,
    "B":    4392.0,
    "V":    5468.0,
    "r'":   6231.0,
    "i'":   7625.0,
    "J":   12350.0,
    "H":   16620.0,
    "Ks":  21590.0,
}

# m_AB = m_input + offset
#
# For Modjaz's procedure:
# - UVW1 is UVOT Vega.
# - UVOT U/B/V were treated as approximately Johnson Vega.
# - FLWO r'/i' are treated as AB.
# - PAIRITEL J/H/Ks are on the 2MASS Vega system.

AB_OFFSET = {
    "UVW2":  1.73,
    "UVM2":  1.69,
    "UVW1":  1.51,

    # Johnson-like values appropriate to the Modjaz description.

    "U":     0.79,
    "B":    -0.09,
    "V":     0.02,

    # Sloan-like magnitudes treated as AB.
    "r'":    0.00,
    "i'":    0.00,

    # 2MASS Vega-to-AB values based on the Cohen zero points.
    "J":     0.894,
    "H":     1.374,
    "Ks":    1.840,
}

STANDARD_FILTERS = [
    "UVW1",
    "U",
    "B",
    "V",
    "r'",
    "i'",
    "J",
    "H",
    "Ks",
]

FAR_UV_FILTERS = ["UVW2", "UVM2"]


DATA_DIR = Path("data")

UV_FILE = DATA_DIR / "SN2008D_UV_M09.txt"
OPT_FILE = DATA_DIR / "SN2008D_OPT_M09.txt"
NIR_FILE = DATA_DIR / "SN2008D_NIR_M09.txt"

DISTANCE_MPC = 31.0
EBV_HOST = 0.60
RV = 3.10

C_AA_S = c.to_value(u.AA / u.s) # speed of light
FNU0_AB = 3631.0e-23  # erg s^-1 cm^-2 Hz^-1 (F_nu0 in AB system)

# Zero-flux endpoints explicitly described by Modjaz.
BLUE_ZERO_AA = 2220.0
RED_ZERO_AA = 24210.0

def a_lambda_f99(wavelength_aa, ebv=EBV_HOST):
    """
    A_lambda in magnitudes for Fitzpatrick (1999).
    """
    av = RV * ebv

    return av * float(
        f99(wavelength_aa * u.AA)
    )


def observed_mag_to_dereddened_ab(
    mag_observed,
    filt,
    ebv=EBV_HOST,
):
    """
    Convert the input magnitude to a dereddened AB magnitude.
    """
    mag_ab_observed = (
        mag_observed
        + AB_OFFSET[filt]
    )

    extinction = a_lambda_f99(
        LAMBDA_EFF[filt],
        ebv=ebv,
    )

    return mag_ab_observed - extinction


def abmag_to_flambda(mag_ab, wavelength_aa):
    """
    AB magnitude -> f_lambda in
    erg s^-1 cm^-2 Angstrom^-1.
    """
    fnu = (
        FNU0_AB
        * 10.0**(-0.4 * mag_ab)
    )

    return (
        fnu
        * C_AA_S
        / wavelength_aa**2
    )


def flux_error_from_mag_error(
    flambda,
    magerr,
    filt,
):
    """
    Propagate statistical and quoted systematic errors.
    """
    statistical = (
        flambda
        * np.log(10.0)
        / 2.5
        * magerr
    )

    systematic_terms = []

    # Modjaz added 0.05 mag systematic uncertainty
    # for UVOT photometry.
    if filt in [
        "UVW2", "UVM2", "UVW1", "U", "B", "V"
    ]:
        uvot_systematic = (
            flambda
            * np.log(10.0)
            / 2.5
            * 0.05
        )

        systematic_terms.append(
            uvot_systematic
        )

    # Modjaz included a 4% flux conversion uncertainty
    # for the Fukugita calibration.
    if filt in ["U", "B", "V", "r'", "i'"]:
        systematic_terms.append(
            0.04 * flambda
        )

    terms = [statistical] + systematic_terms

    return np.sqrt(
        np.sum([
            value**2 for value in terms
        ])
    )


def sed_at_epoch(
    epoch,
    phot,
    include_far_uv=False,
    ebv=EBV_HOST,
):
    filters = list(STANDARD_FILTERS)

    if include_far_uv:
        filters = FAR_UV_FILTERS + filters

    rows = []


    INTERPOLATORS = {
    filt: build_mag_interpolator(group)
    for filt, group in phot.groupby("filter")
    }

    for filt in filters:
        interpolation = INTERPOLATORS.get(filt)

        if interpolation is None:
            continue

        mag = float(
            interpolation["mag"](epoch)
        )

        magerr = abs(float(
            interpolation["magerr"](epoch)
        ))

        mag_ab_dered = (
            observed_mag_to_dereddened_ab(
                mag,
                filt,
                ebv=ebv,
            )
        )

        flambda = abmag_to_flambda(
            mag_ab_dered,
            LAMBDA_EFF[filt],
        )

        flambda_err = flux_error_from_mag_error(
            flambda,
            magerr,
            filt,
        )

        rows.append({
            "time": epoch,
            "filter": filt,
            "wavelength": LAMBDA_EFF[filt],
            "mag_input": mag,
            "magerr": magerr,
            "mag_ab_dered": mag_ab_dered,
            "flambda": flambda,
            "flambda_err": flambda_err,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("wavelength")
        .reset_index(drop=True)
    )


def integrate_sed(
    sed,
    include_far_uv=False,
):
    """
    Direct trapezoidal integration.

    For the standard Modjaz curve, add zero flux at:
        2220 Angstrom
        24210 Angstrom

    The far-UV case is kept separate because the paper's
    1.8-day point extends from UVW2 rather than UVW1.
    """
    if include_far_uv:
        # Approximate blue boundary for a UVW2-through-Ks SED.
        blue_boundary = 1600.0
    else:
        blue_boundary = BLUE_ZERO_AA

    wavelength = sed["wavelength"].to_numpy()
    flambda = sed["flambda"].to_numpy()

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(flambda)
        & (wavelength > blue_boundary)
        & (wavelength < RED_ZERO_AA)
    )

    wavelength = wavelength[valid]
    flambda = flambda[valid]

    wavelength = np.concatenate([
        [blue_boundary],
        wavelength,
        [RED_ZERO_AA],
    ])

    flambda = np.concatenate([
        [0.0],
        flambda,
        [0.0],
    ])

    order = np.argsort(wavelength)

    return np.trapezoid(
        flambda[order],
        wavelength[order],
    )


def read_optical_file(filename):
    """
    Read SN2008D_OPT_M09.txt.

    Only the FLWO r' and i' measurements are used, matching
    the procedure described by Modjaz et al.
    """
    rows = []

    with open(filename, "r") as stream:
        for line in stream:
            if "\t" not in line:
                continue

            columns = [
                value.strip()
                for value in line.rstrip().split("\t")
            ]

            # Expected columns:
            # MJD, Delta_t, B, V, R, R(c), r', i', I, Telescope
            if len(columns) < 10:
                continue

            try:
                time = float(columns[1])
            except ValueError:
                continue

            telescope = columns[9].strip()

            if telescope.upper() != "FLWO":
                continue

            for filt, column_index in [
                ("r'", 6),
                ("i'", 7),
            ]:
                mag, magerr = parse_parenthesis_mag(
                    columns[column_index]
                )

                if not np.isfinite(mag):
                    continue

                rows.append({
                    "time": time,
                    "filter": filt,
                    "mag": mag,
                    "magerr": magerr,
                    "source": "FLWO",
                })

    return pd.DataFrame(rows)

def build_mag_interpolator(group):
    """
    Construct a shape-preserving magnitude interpolator.

    Duplicate times, if present, are combined using their
    inverse-variance weighted mean.
    """
    group = group.copy()

    valid_error = (
        np.isfinite(group["magerr"])
        & (group["magerr"] > 0)
    )

    group["weight"] = np.where(
        valid_error,
        1.0 / group["magerr"]**2,
        1.0,
    )

    grouped = []

    for time, same_time in group.groupby("time"):
        weights = same_time["weight"].to_numpy()
        magnitudes = same_time["mag"].to_numpy()

        mean_mag = np.average(
            magnitudes,
            weights=weights,
        )

        mean_error = np.sqrt(
            1.0 / np.sum(weights)
        )

        grouped.append({
            "time": time,
            "mag": mean_mag,
            "magerr": mean_error,
        })

    grouped = (
        pd.DataFrame(grouped)
        .sort_values("time")
    )

    if len(grouped) < 2:
        return None

    mag_interpolator = PchipInterpolator(
        grouped["time"],
        grouped["mag"],
        extrapolate=True,
    )

    err_interpolator = PchipInterpolator(
        grouped["time"],
        grouped["magerr"],
        extrapolate=True,
    )

    return {
        "mag": mag_interpolator,
        "magerr": err_interpolator,
        "time_min": grouped["time"].min(),
        "time_max": grouped["time"].max(),
    }


def read_nir_file(filename):
    """
    Read SN2008D_NIR_M09.txt.
    """
    rows = []

    with open(filename, "r") as stream:
        for line in stream:
            if "\t" not in line:
                continue

            columns = [
                value.strip()
                for value in line.rstrip().split("\t")
            ]

            # Expected columns:
            # MJD, Delta_t, J, H, Ks
            if len(columns) < 5:
                continue

            try:
                time = float(columns[1])
            except ValueError:
                continue

            for filt, column_index in [
                ("J", 2),
                ("H", 3),
                ("Ks", 4),
            ]:
                mag, magerr = parse_parenthesis_mag(
                    columns[column_index]
                )

                if not np.isfinite(mag):
                    continue

                rows.append({
                    "time": time,
                    "filter": filt,
                    "mag": mag,
                    "magerr": magerr,
                    "source": "PAIRITEL",
                })

    return pd.DataFrame(rows)

def read_uvot_file(filename):
    """
    Read SN2008D_UV_M09.txt.

    Output columns:
        time, filter, mag, magerr, source
    """
    rows = []

    # A valid data line contains:
    # DelT, BinT, Exp, optional ">", mag, optional error, filter
    pattern = re.compile(
        r"^\s*"
        r"([+-]?\d+(?:\.\d+)?)\s+"   # DelT
        r"([+-]?\d+(?:\.\d+)?)\s+"   # BinT
        r"([+-]?\d+(?:\.\d+)?)\s+"   # exposure
        r"(>?)\s*"                    # upper-limit flag
        r"(\d+(?:\.\d+)?)"            # magnitude
        r"(?:\s+(\d+(?:\.\d+)?))?"    # magnitude error
        r"\s+([A-Za-z0-9]+)"           # filter
        r"\s*$"
    )

    with open(filename, "r") as stream:
        for line in stream:
            match = pattern.match(line)

            if match is None:
                continue

            time = float(match.group(1))
            limit_flag = match.group(4)
            mag = float(match.group(5))
            magerr = match.group(6)
            filt = match.group(7)

            # Do not treat upper limits as detections.
            if limit_flag == ">":
                continue

            if magerr is None:
                continue

            rows.append({
                "time": time,
                "filter": filt,
                "mag": mag,
                "magerr": float(magerr),
                "source": "Swift/UVOT",
            })

    return pd.DataFrame(rows)

def parse_parenthesis_mag(value):
    """
    Parse entries such as:
        18.31(04) -> magnitude=18.31, error=0.04
        >20.10    -> upper limit, returned as NaN
        sdots...  -> missing, returned as NaN
        -         -> missing, returned as NaN
    """
    value = str(value).strip().replace(" ", "")

    if (
        not value
        or value.startswith(">")
        or value == "-"
        or "sdot" in value.lower()
    ):
        return np.nan, np.nan

    match = re.fullmatch(
        r"([+-]?\d+(?:\.\d+)?)\((\d+)\)",
        value,
    )

    if match:
        mag_string = match.group(1)
        err_string = match.group(2)

        mag = float(mag_string)

        number_of_decimals = (
            len(mag_string.split(".")[1])
            if "." in mag_string
            else 0
        )

        magerr = (
            float(err_string)
            * 10.0**(-number_of_decimals)
        )

        return mag, magerr

    try:
        return float(value), np.nan
    except ValueError:
        return np.nan, np.nan