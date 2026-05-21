#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TofEstForMACS – Estimate the average neutron path length and time‑of‑flight
for the MACS spectrometer.

The algorithm is a direct translation of the original IDL routine.
It is deliberately written in a clear, step‑by‑step form that
matches the IDL flow, while simultaneously taking advantage of NumPy
vectorisation for speed.

Author(s):
    - Yiming Qiu (original IDL)
    - Richard Azuah (original IDL)
    - Ported to Python by <your‑name>

Created: 2021‑07 (IDL) Ported: 2024‑06
"""

from __future__ import annotations

import time as _time
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import moment


SEED=12

# -------------------------------------------------------------
# Helper – tiny wrapper to keep the IDL‑style `print` output
# -------------------------------------------------------------
def _prt(*args: Any, **kwargs: Any) -> None:
    """Print with a trailing newline (just a thin alias for built‑in print)."""
    print(*args, **kwargs)


# -------------------------------------------------------------
# Main routine
# -------------------------------------------------------------
def TofEstForMACS(
    Ef: float | None = None,
    *,
    w_sample: float | None = None,
    h_sample: float | None = None,
    nPoints: int | None = None,
    noplot: bool = False,
) -> Dict[str, Any]:
    """
    Estimate mean neutron path length and TOF for MACS.

    Parameters
    ----------
    Ef : float, optional
        Final neutron energy in meV.  If *None* the default (5.0 meV) is used.
    w_sample : float, optional
        Sample width (or diameter for a cylinder) in **cm**.
        If not supplied a default width of 1 cm is used.
    h_sample : float, optional
        Sample height in **cm**.  If not supplied a default height of 2 cm
        is used.
    nPoints : int, optional
        Number of random points to generate inside each component.
        The original IDL used 31 as the default.
    noplot : bool, default=False
        If *True* the routine skips the Matplotlib histogram/TOF plots.

    Returns
    -------
    dict
        Dictionary with the most important numerical results, e.g.
        ``{'Lsd_mean_m': …, 'Lsd_sigma_m': …, 'tof_mean_ms': …,
          'tof_sigma_ms': …, 'hist_bins': …, 'hist_counts': …}``.

    Notes
    -----
    The implementation follows the IDL logic step‑by‑step, but the
    expensive four‑nested‑loop distance accumulation is replaced by a fully
    vectorised NumPy computation.  For the default `nPoints=31` the vectorised
    version creates an array of size `(31, 31, 31, 31)` → 923 521 elements,
    which comfortably fits in memory on a modern laptop.
    """

    # ---------------------------------------------------------
    # 0.  Argument handling – mimic IDL's “if N_elements()==0 …”
    # ---------------------------------------------------------
    if Ef is None:
        Ef = 5.0  # default 5.0 meV
    if w_sample is None:
        w = 1.0 / 100.0            # 1 cm → 0.01 m
    else:
        w = w_sample / 100.0       # cm → m
    if h_sample is None:
        h = 2.0 / 100.0            # 2 cm → 0.02 m
    else:
        h = h_sample / 100.0       # cm → m
    if nPoints is None:
        nPoints = 31

    # ---------------------------------------------------------
    # 1.  Fixed instrument geometry (identical to IDL constants)
    # ---------------------------------------------------------
    Lsa = 1.00       # sample‑to‑analyzer distance (m)   (a6 = 90°)
    Laa = 0.07       # analyzer‑to‑analyzer distance (m)
    Lad = 0.25       # analyzer‑to‑detector distance (m)

    # scattering angle a6 (radians) – the same empirical formula as IDL
    a6 = np.arcsin(np.sqrt(81.8042 / Ef) / 2.0 / 3.35416) * 2

    # neutron velocity (m s⁻¹) from Ef (meV)
    v = np.sqrt(Ef / 5.22704e-6)

    # empirical energy spread for a Be filter (valid for Ef < 5 meV)
    dEf = -0.188852 + 0.107677 * Ef
    dv = v / (2.0 * Ef) * dEf

    # central beam path length (m) and corresponding flight time (s)
    L0 = Lsa + Lad + Laa / np.sin(a6) - Laa / np.tan(a6)
    time_center = L0 / v

    # ---------------------------------------------------------
    # 2.  Component size limits (x, y, z) --------------------
    # ---------------------------------------------------------

    # ----- Sample (cylindrical) ---------------------------------
    s_x_min, s_x_max = -w / 2.0, w / 2.0
    s_y_min, s_y_max = s_x_min, s_x_max
    s_z_min, s_z_max = -h / 2.0, h / 2.0

    # ----- Analyzer 1 ------------------------------------------
    w_a = 0.06                         # width 6 cm → 0.06 m
    # height is defined via a curvature; keep the same expression
    h_a = 0.5 * np.sin(np.arcsin(1.0 / 50.0) * 9.0) * 2   # ≈0.18 m

    a1_x_min = -w_a / 2.0 * np.sin(a6 / 2.0)
    a1_x_max =  w_a / 2.0 * np.sin(a6 / 2.0)

    a1_y_center = Lsa - Laa / np.tan(a6) / 2.0
    a1_y_min = a1_y_center - w_a / 2.0 * np.cos(a6 / 2.0)
    a1_y_max = a1_y_center + w_a / 2.0 * np.cos(a6 / 2.0)

    a1_z_min, a1_z_max = -h_a / 2.0, h_a / 2.0

    # ----- Analyzer 2 (identical height, shifted in x‑y) -------
    a2_x_min = Laa - w_a / 2.0 * np.sin(a6 / 2.0)
    a2_x_max = Laa + w_a / 2.0 * np.sin(a6 / 2.0)

    a2_y_center = Lsa + Laa / np.tan(a6) / 2.0
    a2_y_min = a2_y_center - w_a / 2.0 * np.cos(a6 / 2.0)
    a2_y_max = a2_y_center + w_a / 2.0 * np.cos(a6 / 2.0)

    a2_z_min, a2_z_max = a1_z_min, a1_z_max

    # ----- Detector ---------------------------------------------
    h_d = 0.14            # height (m)
    r_d = 0.013           # radius (m)

    d_x_min, d_x_max = Laa - r_d, Laa + r_d
    d_y_min, d_y_max = Lsa + Lad - r_d, Lsa + Lad + r_d
    d_z_min, d_z_max = -h_d / 2.0, h_d / 2.0

    # ---------------------------------------------------------
    # 3.  Random point generation – uniform distribution
    # ---------------------------------------------------------
    rng = np.random.default_rng(seed=SEED)   # uses a fresh entropy source each call

    # Sample points
    s_x = rng.uniform(s_x_min, s_x_max, nPoints)
    s_y = rng.uniform(s_y_min, s_y_max, nPoints)
    s_z = rng.uniform(s_z_min, s_z_max, nPoints)

    # Analyzer‑1 points
    a1_x = rng.uniform(a1_x_min, a1_x_max, nPoints)
    a1_y = rng.uniform(a1_y_min, a1_y_max, nPoints)
    a1_z = rng.uniform(a1_z_min, a1_z_max, nPoints)

    # Analyzer‑2 points
    a2_x = rng.uniform(a2_x_min, a2_x_max, nPoints)
    a2_y = rng.uniform(a2_y_min, a2_y_max, nPoints)
    a2_z = rng.uniform(a2_z_min, a2_z_max, nPoints)

    # Detector points
    d_x = rng.uniform(d_x_min, d_x_max, nPoints)
    d_y = rng.uniform(d_y_min, d_y_max, nPoints)
    d_z = rng.uniform(d_z_min, d_z_max, nPoints)

    # ---------------------------------------------------------
    # 4.  Build all possible 4‑tuples and compute total path
    # ---------------------------------------------------------
    _prt("\nStarting distance calculation – this may take a few seconds…")
    start = _time.perf_counter()

    # We reshape each coordinate array to expose a new axis that will be
    # broadcast against the others:
    #
    #   s_? : (ns, 1, 1, 1)
    #   a1_?: (1, na, 1, 1)
    #   a2_?: (1, 1, na, 1)
    #   d_? : (1, 1, 1, nd)
    #
    # The final distance array therefore has shape (ns, na, na, nd).

    s_x4 = s_x[:, None, None, None]
    s_y4 = s_y[:, None, None, None]
    s_z4 = s_z[:, None, None, None]

    a1_x4 = a1_x[None, :, None, None]
    a1_y4 = a1_y[None, :, None, None]
    a1_z4 = a1_z[None, :, None, None]

    a2_x4 = a2_x[None, None, :, None]
    a2_y4 = a2_y[None, None, :, None]
    a2_z4 = a2_z[None, None, :, None]

    d_x4 = d_x[None, None, None, :]
    d_y4 = d_y[None, None, None, :]
    d_z4 = d_z[None, None, None, :]

    # Vectorised Euclidean distances
    L1 = np.sqrt((a1_x4 - s_x4) ** 2 + (a1_y4 - s_y4) ** 2 + (a1_z4 - s_z4) ** 2)
    L2 = np.sqrt((a2_x4 - a1_x4) ** 2 + (a2_y4 - a1_y4) ** 2 + (a2_z4 - a1_z4) ** 2)
    L3 = np.sqrt((d_x4 - a2_x4) ** 2 + (d_y4 - a2_y4) ** 2 + (d_z4 - a2_z4) ** 2)

    # Total sample‑detector path for every combination
    Lsd = L1 + L2 + L3               # shape (ns, na, na, nd)

    # Flatten to a 1‑D array – this is exactly what the IDL code did by
    # repeatedly appending to a list.
    Lsd = Lsd.ravel()

    elapsed = _time.perf_counter() - start
    _prt(f"  Done.  {Lsd.size:,} path lengths computed in {elapsed:.2f} s.\n")
    _prt(f"  Number of elements in Lsd = {Lsd.size}")

    # ---------------------------------------------------------
    # 5.  Basic statistics -------------------------------------------------
    # ---------------------------------------------------------
    dL = np.max(np.abs(Lsd - L0))
    d_time = time_center * np.sqrt((dL / L0) ** 2 + (dv / v) ** 2)

    _prt(f"Ef = {Ef:.2f} meV,  time = {time_center:.6f} ± {d_time:.6f}  secs")

    # Moment (mean, variance) – SciPy moment with order 0 & 2 yields
    # mean and central 2nd moment (variance) when `moment(..., moment=2)`.
    Lsd_mean = np.mean(Lsd)                     # <L>
    Lsd_var  = np.var(Lsd, ddof=0)               # σ² (population)
    Lsd_sigma = np.sqrt(Lsd_var)

    # TOF statistics (mean & sigma) = L/v
    tof_mean = Lsd_mean / v                     # seconds
    tof_sigma = Lsd_sigma / v

    # -----------------------------------------------------------------
    # 6.  Print nicely formatted results (mirrors IDL output)
    # -----------------------------------------------------------------
    _prt("\n##################################################")
    _prt("*** MACS ***")
    _prt("##################################################")
    fmt1 = "6.1f"
    fmt2 = "6.3f"
    # -- strings that were built with IDL's `String` routine
    results = [
        f"Ef = {Ef:{fmt2}} meV",
        f"Mean S-D dist = {Lsd_mean * 1000:{fmt1}} mm",
        f"σ of S-D dist = {Lsd_sigma * 1000:{fmt1}} mm",
        f"Mean TOF      = {tof_mean * 1000:{fmt2}} ms",
        f"σ of TOF      = {tof_sigma * 1000:{fmt2}} ms",
        f"Δt/t          = {(tof_sigma / tof_mean) * 100:{'4.1f'}} %",
    ]
    for line in results:
        _prt(line)
    _prt("##################################################\n")

    # ---------------------------------------------------------
    # 7.  Histogram + optional Matplotlib plots
    # ---------------------------------------------------------
    if not noplot:
        # 51 bins is the same number used by IDL's HISTOGRAM keyword.
        nbins = 51
        counts, bin_edges = np.histogram(Lsd, bins=nbins)
        # Bin centres for nicer plotting (IDL used the `locations=` keyword)
        distances = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        times = distances / v

        # --- Plot 1: distance histogram -----------------------
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(distances, counts, "d", color="tab:blue", linestyle="none")
        ax1.set_xlabel(r"$L_{\mathrm{samp-detector}}$ (m)")
        ax1.set_ylabel("Intensity")
        ax1.set_title(f"MACS: Lsd Estimates for Ef = {Ef:.2f} meV")
        # embed the numeric results in the figure (similar to IDL's TEXT)
        txt = "\n".join(results[:3])
        ax1.text(0.5, 0.7, txt, transform=ax1.transAxes,
                 fontsize=10, fontfamily="monospace", va="top",
                 ha="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        fig1.tight_layout()

        # --- Plot 2: TOF histogram ----------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(times, counts, "d", color="tab:blue", linestyle="none")
        ax2.set_xlabel(r"$\mathrm{TOF}_{\mathrm{samp-detector}}$ (s)")
        ax2.set_ylabel("Intensity")
        ax2.set_title(f"MACS: TOF Estimates for Ef = {Ef:.2f} meV")
        txt2 = "\n".join([results[0]] + results[3:])   # use the TOF‑related lines
        ax2.text(0.5, 0.7, txt2, transform=ax2.transAxes,
                 fontsize=10, fontfamily="monospace", va="top",
                 ha="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        fig2.tight_layout()

        plt.show()

    # ---------------------------------------------------------
    # 8.  Return a dictionary with the quantitative results
    # ---------------------------------------------------------
    return {
        "Ef_meV": Ef,
        "L0_m": L0,
        "time_center_s": time_center,
        "Lsd_mean_m": Lsd_mean,
        "Lsd_sigma_m": Lsd_sigma,
        "tof_mean_s": tof_mean,
        "tof_sigma_s": tof_sigma,
        "relative_t_error": (tof_sigma / tof_mean) * 100.0,
        "Lsd_array_m": Lsd,          # full array – keep it if the caller wants it
        "hist_counts": counts,
        "hist_bin_edges_m": bin_edges,
        "hist_bin_centers_m": distances,
        "hist_bin_centers_s": times,
    }


# -----------------------------------------------------------------
# Simple test driver – mimics a little IDL interactive session
# -----------------------------------------------------------------
if __name__ == "__main__":
    # Example reproduced from the IDL documentation:
    #   Ef = 5 meV, default geometry, 31 points, plot enabled.
    result = TofEstForMACS(Ef=5.0, w_sample=1.0, h_sample=2.0, nPoints=31, noplot=False)

    # The returned dictionary can be inspected programmatically:
    _prt("\nReturned dictionary keys:", list(result.keys()))