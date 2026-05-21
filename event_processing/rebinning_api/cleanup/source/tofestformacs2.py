#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TofEstForMACS
--------------

Simple program to estimate the average neutron paths and time‑of‑flight
on MACS.

This is done by generating random points within the components along the
flight path (sample, analyzers and detector) and then calculating the total
distances between these points and then making a statistical analysis on
these to determine their mean and standard deviation in order to calculate
the mean neutron tof and standard deviation. This estimate is simplistic
and only takes into account the physical extent of the components and
nothing else.

Keywords (input arguments)
--------------------------
Ef        : float, final neutron energy in meV (default 5.0 meV)
w_sample  : float, sample width in cm (or diameter if cylindrical)
h_sample  : float, sample height in cm
nPoints   : int,   number of random points per component (default 31)
noplot    : bool,  suppress plotting when True
dx, dy, dz: (unused) kept for compatibility with the original signature
"""

SEED = 12
# ======================================================================
# Imports & global constants ------------------------------------------------
# ======================================================================
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import moment   # alternative: np.mean, np.var
from time import perf_counter   # IDL's Tic/Toc equivalent

# ----------------------------------------------------------------------
# Helper functions that mimic a few IDL behaviours
# ----------------------------------------------------------------------
def moment0(arr):
    """Return the 0‑th (mean) and 1‑st (variance) moments of an array."""
    # IDL's Moment returns an array where [0] = mean, [1] = variance
    return np.mean(arr), np.var(arr, ddof=0)   # ddof=0 → population variance

def histogram(arr, nbins=51):
    """Return histogram counts and bin centres (like IDL's HISTOGRAM)."""
    counts, edges = np.histogram(arr, bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return counts, centres

# ======================================================================
# Main routine -----------------------------------------------------------
# ======================================================================
def tof_est_for_macs(
    Ef=None,
    w_sample=None,
    h_sample=None,
    nPoints=None,
    noplot=False,
    dx=None, dy=None, dz=None   # kept for signature compatibility
):
    """
    Translate of IDL pro TofEstForMACS.
    All arguments are optional – they follow the same defaults as the IDL code.
    """
    # ------------------------------------------------------------------
    # if N_elements(Ef) eq 0 then Ef = 5.0   ; default 5.0 meV
    # ------------------------------------------------------------------
    if Ef is None:
        Ef = 5.0                     # default value, same as IDL
    
    # ------------------------------------------------------------------
    # if N_elements(w_sample) eq 0 then w = 1/100. else w = w_sample/100.
    #   ;cm->m
    # ------------------------------------------------------------------
    if w_sample is None:
        w = 1.0 / 100.0              # 1 cm → 0.01 m
    else:
        w = w_sample / 100.0         # conversion from cm to m
    
    # ------------------------------------------------------------------
    # if N_elements(h_sample) eq 0 then h = 2/100. else h = h_sample/100.
    # ------------------------------------------------------------------
    if h_sample is None:
        h = 2.0 / 100.0
    else:
        h = h_sample / 100.0
    
    # ------------------------------------------------------------------
    # if N_elements(nPoints) eq 0 then nPoints = 31
    # ------------------------------------------------------------------
    if nPoints is None:
        nPoints = 31
    
    # ------------------------------------------------------------------
    # Lsa = 1.00 ; sample‑to‑analyzer distance (m) when a6 = 90°
    # Laa = 0.07 ; analyzer‑to‑analyzer distance (m) when a6 = 90°
    # Lad = 0.25 ; analyzer‑to‑detector distance (m) when a6 = 90°
    # ------------------------------------------------------------------
    Lsa = 1.00
    Laa = 0.07
    Lad = 0.25
    
    # ------------------------------------------------------------------
    # a6 = Asin( Sqrt(81.8042/Ef) / 2.0 / 3.35416 ) * 2   ; Ef in meV
    # ------------------------------------------------------------------
    a6 = np.arcsin(np.sqrt(81.8042 / Ef) / (2.0 * 3.35416)) * 2.0
    
    # ------------------------------------------------------------------
    # v = Sqrt(Ef / 5.22704e-6)   ; neutron velocity (m/s) from Ef in meV
    # ------------------------------------------------------------------
    v = np.sqrt(Ef / 5.22704e-6)
    
    # ------------------------------------------------------------------
    # dEf = -0.188852 + 0.107677*Ef   ; empirical formula for Ef<5 meV with Be filter
    # ------------------------------------------------------------------
    dEf = -0.188852 + 0.107677 * Ef
    
    # ------------------------------------------------------------------
    # dv = v/(2.*Ef) * dEf
    # ------------------------------------------------------------------
    dv = v / (2.0 * Ef) * dEf
    
    # ------------------------------------------------------------------
    # L0 = (Lsa+Lad+Laa/Sin(a6)-Laa/Tan(a6)) ; travel distance for beam centre
    # ------------------------------------------------------------------
    L0 = (Lsa + Lad + Laa / np.sin(a6) - Laa / np.tan(a6))
    
    # ------------------------------------------------------------------
    # time = L0/v   ; in seconds
    # ------------------------------------------------------------------
    time = L0 / v
    
    # ==================================================================
    # Geometry limits (min/max) for each component and random point generation
    # ==================================================================
    
    # ------------------------------------------------------------------
    # analyzer dimension 6 cm (width) x 2 cm (ignore 2 mm thickness) x 9,
    #    in curvature of 50 cm
    # ------------------------------------------------------------------
    w_a = 0.06                                             # m
    
    # ------------------------------------------------------------------
    # h_a = 0.5*Sin( Asin(1./50)*9 )   ; ~0.18 m
    # ------------------------------------------------------------------
    h_a = 0.5 * np.sin(np.arcsin(1.0 / 50.0) * 9.0)*2
    
    # ------------------------------------------------------------------
    # Sample coordinates, assuming cylindrical shape
    # s_x_min = -w/2.0 & s_x_max = w/2.0
    # s_y_min = s_x_min & s_y_max = s_x_max
    # s_z_min = -h/2.0 & s_z_max = h/2.0
    # ------------------------------------------------------------------
    s_x_min, s_x_max = -w / 2.0,  w / 2.0
    s_y_min, s_y_max = s_x_min,  s_x_max
    s_z_min, s_z_max = -h / 2.0, h / 2.0
    
    ns = nPoints
    
    # ------------------------------------------------------------------
    # IDL's Randomu(seed, n) returns uniform numbers in [0,1)
    # In Python we just use numpy.random.default_rng().uniform()
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=SEED)
    s_x = rng.uniform(s_x_min, s_x_max, ns)
    s_y = rng.uniform(s_y_min, s_y_max, ns)
    s_z = rng.uniform(s_z_min, s_z_max, ns)
    
    # ------------------------------------------------------------------
    # Analyzer‑1 coordinates
    # ------------------------------------------------------------------
    a1_x_min = -w_a / 2.0 * np.sin(a6 / 2.0)
    a1_x_max =  w_a / 2.0 * np.sin(a6 / 2.0)
    a1_y_min = Lsa - Laa / np.tan(a6) / 2.0 - w_a / 2.0 * np.cos(a6 / 2.0)
    a1_y_max = Lsa - Laa / np.tan(a6) / 2.0 + w_a / 2.0 * np.cos(a6 / 2.0)
    a1_z_min = -h_a / 2.0
    a1_z_max =  h_a / 2.0
    
    na = nPoints
    a1_x = rng.uniform(a1_x_min, a1_x_max, na)
    a1_y = rng.uniform(a1_y_min, a1_y_max, na)
    a1_z = rng.uniform(a1_z_min, a1_z_max, na)
    
    # ------------------------------------------------------------------
    # Analyzer‑2 coordinates
    # ------------------------------------------------------------------
    a2_x_min = Laa - w_a / 2.0 * np.sin(a6 / 2.0)
    a2_x_max = Laa + w_a / 2.0 * np.sin(a6 / 2.0)
    a2_y_min = Lsa + Laa / np.tan(a6) / 2.0 - w_a / 2.0 * np.cos(a6 / 2.0)
    a2_y_max = Lsa + Laa / np.tan(a6) / 2.0 + w_a / 2.0 * np.cos(a6 / 2.0)
    a2_z_min = a1_z_min
    a2_z_max = a1_z_max
    
    a2_x = rng.uniform(a2_x_min, a2_x_max, na)
    a2_y = rng.uniform(a2_y_min, a2_y_max, na)
    a2_z = rng.uniform(a2_z_min, a2_z_max, na)
    
    # ------------------------------------------------------------------
    # Detector coordinates, detector height 14 cm, radius 1.3 cm
    # ------------------------------------------------------------------
    h_d = 0.14                 # m
    r_d = 0.013                # m
    
    d_x_min = Laa - r_d
    d_x_max = Laa + r_d
    d_y_min = Lsa + Lad - r_d
    d_y_max = Lsa + Lad + r_d
    d_z_min = -h_d / 2.0
    d_z_max =  h_d / 2.0
    
    nd = nPoints
    d_x = rng.uniform(d_x_min, d_x_max, nd)
    d_y = rng.uniform(d_y_min, d_y_max, nd)
    d_z = rng.uniform(d_z_min, d_z_max, nd)
    
    # ------------------------------------------------------------------
    # Tic/Toc – measurement of elapsed CPU time
    # ------------------------------------------------------------------
    t_start = perf_counter()
    
    # ==================================================================
    # Distance calculation – nested loops (exactly as in the IDL code)
    # ==================================================================
    # NOTE: This is O(ns·na·na·nd) and can be very slow.
    # For a production version you would vectorise it, but we keep the
    # explicit loops here to mirror the IDL logic for easy checking.
    # ==================================================================
    Lsd = []          # list that will store each total path length
    
    for i in range(ns):
        for j in range(na):
            for k in range(na):
                for l in range(nd):
                    # L1 = sqrt((a1_x[j]-s_x[i])^2 + (a1_y[j]-s_y[i])^2 + (a1_z[j]-s_z[i])^2) +
                    #      sqrt((a2_x[k]-a1_x[j])^2 + (a2_y[k]-a1_y[j])^2 + (a2_z[k]-a1_z[j])^2) +
                    #      sqrt((d_x[l]-a2_x[k])^2 + (d_y[l]-a2_y[k])^2 + (d_z[l]-a2_z[k])^2)
                    L1 = np.sqrt(
                        (a1_x[j] - s_x[i])**2 +
                        (a1_y[j] - s_y[i])**2 +
                        (a1_z[j] - s_z[i])**2
                    )
                    L1 += np.sqrt(
                        (a2_x[k] - a1_x[j])**2 +
                        (a2_y[k] - a1_y[j])**2 +
                        (a2_z[k] - a1_z[j])**2
                    )
                    L1 += np.sqrt(
                        (d_x[l] - a2_x[k])**2 +
                        (d_y[l] - a2_y[k])**2 +
                        (d_z[l] - a2_z[k])**2
                    )
                    Lsd.append(L1)
    
    t_end = perf_counter()
    print(f"Distance calculation time: {t_end - t_start:.3f} s")
    
    # ------------------------------------------------------------------
    # Print number of generated distances (ID: Print, N_elements(Lsd))
    # ------------------------------------------------------------------
    print(f"N_elements(Lsd) = {len(Lsd)}")
    
    # ------------------------------------------------------------------
    # dL = Max(Abs(Lsd-L0))
    # d_time = time * Sqrt( (dL/L0)^2 + (dv/v)^2 )
    # ------------------------------------------------------------------
    Lsd_arr = np.array(Lsd)
    dL = np.max(np.abs(Lsd_arr - L0))
    d_time = time * np.sqrt( (dL / L0)**2 + (dv / v)**2 )
    
    print(f"Ef = {Ef:.3f} meV, time = {time:.6e} +/- {d_time:.6e} secs")
    
    # ==================================================================
    # Statistical analysis – same output format as IDL
    # ==================================================================
    # moments = Moment(Lsd)   →  [mean, variance]
    # ------------------------------------------------------------------
    mean_Lsd, var_Lsd = moment0(Lsd_arr)
    sigma_Lsd = np.sqrt(var_Lsd)
    
    # ------------------------------------------------------------------
    # Formatting strings (IDL used format='(F6.1)', '(F6.3)', etc.)
    # ------------------------------------------------------------------
    fmt1 = "6.1f"
    fmt2 = "6.3f"
    
    results = []
    results.append(f"Ef = {Ef:{fmt1}} meV")
    results.append(f"Mean S-D dist = {mean_Lsd*1000.0:{fmt1}} mm")
    results.append(f"σ of S-D dist = {sigma_Lsd*1000.0:{fmt1}} mm")
    results.append(f"Mean TOF      = {mean_Lsd/v*1000.0:{fmt2}} ms")
    results.append(f"σ of TOF      = {sigma_Lsd/v*1000.0:{fmt2}} ms")
    results.append(f"Δt/t          = {sigma_Lsd/mean_Lsd*100.0:4.1f} %")
    
    print("\n##################################################")
    print("*** MACS ***")
    print("##################################################")
    for line in results:
        print(line)
    print("##################################################")
    
    # ------------------------------------------------------------------
    # If noplot keyword is set → exit early (same as IDL's Keyword_set)
    # ------------------------------------------------------------------
    if noplot:
        return
    
    # ==================================================================
    # Histogram & plotting (optional – reproduced exactly as IDL)
    # ==================================================================
    # NOTE: In the original IDL code the variable `distances` is filled
    # automatically by the HISTOGRAM call. In Python we return it explicitly.
    # ------------------------------------------------------------------
    frequency, distances = histogram(Lsd_arr, nbins=51)
    times = distances / v
    
    # ---- Plot 1: distance distribution ---------------------------------
    xtitle = r"$L_{samp-detector}$ (m)"
    ytitle = "Intensity"
    title  = f"MACS: Lsd Estimates for Ef = {Ef:.2f} meV"
    plt.figure(figsize=(10, 6))
    plt.plot(distances, frequency, linestyle=' ', marker='D', color='blue')
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    # Add the text block (results[0:3]) in the same location as IDL's Text
    txt = "\n".join(results[0:3])
    plt.text(0.5, 0.7, txt, transform=plt.gca().transAxes,
             fontsize=10, fontfamily='Courier', verticalalignment='top')
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()
    
    # ---- Plot 2: TOF distribution --------------------------------------
    xtitle = r"$TOF_{samp-detector}$ (s)$"
    title  = f"MACS: TOF Estimates for Ef = {Ef:.2f} meV"
    plt.figure(figsize=(10, 6))
    plt.plot(times, frequency, linestyle=' ', marker='D', color='blue')
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    txt = "\n".join(results[0:1] + results[3:6])   # results[0] plus 3‑5
    plt.text(0.5, 0.7, txt, transform=plt.gca().transAxes,
             fontsize=10, fontfamily='Courier', verticalalignment='top')
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()
    
    # End of routine ----------------------------------------------------
    
# ----------------------------------------------------------------------
# If the script is executed directly, run a quick demo with the default
# parameters (mirrors the behavior of the IDL program when called without
# arguments).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple command‑line interface (optional)
    import argparse
    parser = argparse.ArgumentParser(description="Estimate neutron TOF for MACS")
    parser.add_argument("-E", "--Ef", type=float, help="Final neutron energy (meV)")
    parser.add_argument("-w", "--w_sample", type=float, help="Sample width (cm)")
    parser.add_argument("-l", "--h_sample", type=float, help="Sample height (cm)")
    parser.add_argument("-n", "--nPoints", type=int, help="Number of random points per component")
    parser.add_argument("-p", "--noplot", action="store_true", help="Suppress plotting")
    args = parser.parse_args()
    
    tof_est_for_macs(
        Ef=args.Ef,
        w_sample=args.w_sample,
        h_sample=args.h_sample,
        nPoints=args.nPoints,
        noplot=args.noplot
    )