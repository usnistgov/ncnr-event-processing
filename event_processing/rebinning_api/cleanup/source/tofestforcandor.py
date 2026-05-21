#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
;+
; NAME:
; TofEstForCANDOR
;
; PURPOSE:
; Simple program to estimate the average neutron paths and time-of-flight
; on CANDOR.
;
; This is done by generating random points within the components along
; the flight path (sample, HOPG analyzer and detector) and then calculating
; the total distances between these points and then making a statistical
; analysis on these to determine their mean and standard deviation in
; order to calculate the mean neutron tof and standard deviation. This
; estimate is simplistic and only takes into account the physical extent
; of the components and nothing else.
;
; PARAMETERS
;  wavelength - neutron wavelength
;
; KEYWORDS:
;  SampleDim   - Sample dimensions [Length,Width,thickness] where Length is parallel to the beam in mm
;  AnalyzerDim - HOPG dimensions [width, Height, thickness] in mm
;  DetectorDim - Detector dimensions [width, Height, thickness] in mm
;  thetaS      - the specular angle of the beam on the sample (for detector bank0) in degrees
;  thetaA      - HOPG theta in degrees
;  Lsa         - Sample-HOPG distance in mm
;  Lad         - HOPG-Detector distance in mm
;  nPoints     - the number of points to generate within each of the components
;  noPlotFlag  - set to disable plotting of the results
;
; AUTHORS:
;  Richard Azuah
;  July, 2021
;-
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from typing import Any, Tuple, List, Optional

# ----------------------------------------------------------------------
# Helper functions (tiny wrappers that mimic a few IDL behaviours)
# ----------------------------------------------------------------------
def moment(arr: np.ndarray) -> Tuple[float, float]:
    """
    Return (mean, variance) – the same 2‑element vector that IDL’s MOMENT()
    returns with default arguments.
    """
    mean = np.mean(arr)
    var  = np.var(arr, ddof=0)      # population variance → matches IDL
    return mean, var


def histogram(arr: np.ndarray, nbins: int = 51) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (frequency, bin_centers) – equivalent to IDL's
    HISTOGRAM(arr, nbins=..., locations=distances).
    """
    freq, edges = np.histogram(arr, bins=nbins, density=False)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return freq, centers


# ----------------------------------------------------------------------
# Main routine – translates the IDL procedure `TofEstForCANDOR`
# ----------------------------------------------------------------------
def ToFEstForCANDOR(
    wavelength: float,
    DeltaLambda: float,
    SampleDim: Optional[List[float]] = None,
    thetaS: Optional[float] = None,
    Lsa: Optional[float] = None,
    AnalyzerDim: Optional[List[float]] = None,
    DetectorDim: Optional[List[float]] = None,
    thetaA: Optional[float] = None,
    Lad: Optional[float] = None,
    nPoints: int = 0,
    noPlotFlag: bool = False,
    **_extra: Any,
) -> Tuple[np.ndarray, dict]:
    """
    Estimate the average neutron path length and TOF for the CANDOR
    instrument, using the same Monte‑Carlo scheme as the original IDL code.
    """

    # ------------------------------------------------------------------
    # Set some default values if needed
    # Default distances are for the first (upstream) analyzer/detector in Bank0!
    # ------------------------------------------------------------------
    if wavelength is None:
        wavelength = 6.00                     # mean wavelength in angstroms
    if SampleDim is None or len(SampleDim) != 3:
        SampleDim = [50.0, 10.0, 5.0]          # sample [Length, Width, thickness] (mm)
    if thetaS is None:
        thetaS = 10.0                          # specular angle for detector bank0 (deg)
    if AnalyzerDim is None or len(AnalyzerDim) != 3:
        AnalyzerDim = [12.0, 30.0, 1.0]        # HOPG [width, height, thickness] (mm)
    if DetectorDim is None or len(DetectorDim) != 3:
        DetectorDim = [6.0, 30.0, 2.0]         # Detector [width, height, thickness] (mm)
    if Lsa is None:
        Lsa = 3572.81                          # Sample to the HOPD in question (mm)
    if Lad is None:
        Lad = 18.100                           # HOPD to its accompanying detector (mm)
    if thetaA is None:
        thetaA = 63.1                          # specular angle for detector bank0 (deg)
    if nPoints == 0:
        nPoints = 101

    # ------------------------------------------------------------------
    # Convert to radians and metres (same as IDL)
    # ------------------------------------------------------------------
    deg2rad = np.pi / 180.0
    thetaS = thetaS * deg2rad                 # to radians
    thetaA = thetaA * deg2rad

    SampleDim   = np.asarray(SampleDim)   * 0.001   # mm → m
    AnalyzerDim = np.asarray(AnalyzerDim) * 0.001
    DetectorDim = np.asarray(DetectorDim) * 0.001
    Lsa = Lsa * 0.001
    Lad = Lad * 0.001

    hOverMn = 3956.034                      # tof = d*wavelength/hOverMn

    # ######################################################################
    # Based on the geometry of the instrument, determine (min, max) limits
    # along the x, y and z axes for each of the components
    # Then, generate random points that lie within the set limits.
    #
    # Assume a coordinate system where
    #   y is along the scattered beam (after sample)
    #   z is into the paper (along sample width)
    #   x is pointing up along height of HOPG/Detector
    # Hence the incident and scattered beam on the sample lie within the x‑y plane
    # ######################################################################
    ns = nPoints
    Ls, Ws, Ts = SampleDim                     # Sample is assumed to have dimensions Len,Width,Thickness

    # ----- Sample limits -------------------------------------------------
    xMin = -Ts/2.0*np.cos(thetaS) - Ls/2.0*np.sin(thetaS)
    xMax =  Ts/2.0*np.cos(thetaS) + Ls/2.0*np.sin(thetaS)
    SxCoords = np.random.random(ns)*(xMax - xMin) + xMin   # generate ns random values between xMin and xMax

    yMin = -Ts/2.0*np.sin(thetaS) - Ls/2.0*np.cos(thetaS)
    yMax =  Ts/2.0*np.sin(thetaS) + Ls/2.0*np.cos(thetaS)
    SyCoords = np.random.random(ns)*(yMax - yMin) + yMin   # generate ns random values between yMin and yMax

    zMin = -Ws/2.0
    zMax =  Ws/2.0
    SzCoords = np.random.random(ns)*(zMax - zMin) + zMin   # generate ns random values between zMin and zMax

    # ----- HOPG / analyzer limits ---------------------------------------
    na = nPoints
    Wa, Ha, Ta = AnalyzerDim                     # HOPG has dimensions Width (Wa), Height (Ha) and thickness (Ta)

    xMin = -Ha/2.0
    xMax =  Ha/2.0
    AxCoords = np.random.random(na)*(xMax - xMin) + xMin   # generate na random values between xMin and xMax

    yMin = Lsa - Wa/2.0*np.cos(thetaA)
    yMax = Lsa + Wa/2.0*np.cos(thetaA)
    AyCoords = np.random.random(na)*(yMax - yMin) + yMin   # generate na random values between yMin and yMax

    zMin = -Wa/2.0*np.sin(thetaA) - Ta/2.0*np.cos(thetaA)
    zMax =  Wa/2.0*np.sin(thetaA) + Ta/2.0*np.cos(thetaA)
    AzCoords = np.random.random(na)*(zMax - zMin) + zMin   # generate na random values between zMin and zMax

    # ----- Scintillator detector limits ---------------------------------
    nd = nPoints
    Wd, Hd, Td = DetectorDim                     # Detector has dimensions Width (Wd), Height (Hd) and thickness (Td)

    xMin = -Hd/2.0
    xMax =  Hd/2.0
    DxCoords = np.random.random(nd)*(xMax - xMin) + xMin   # generate nd random values between xMin and xMax

    yMin = Lsa + Lad*np.cos(2*thetaA) - Wd/2.0
    yMax = Lsa + Lad*np.cos(2*thetaA) + Wd/2.0
    DyCoords = np.random.random(nd)*(yMax - yMin) + yMin   # generate nd random values between yMin and yMax

    zMin = Lad*np.sin(2*thetaA) - Td/2.0
    zMax = Lad*np.sin(2*thetaA) + Td/2.0
    DzCoords = np.random.random(nd)*(zMax - zMin) + zMin   # generate nd random values between zMin and zMax

    # ######################################################################
    # Now calculate the distance between the components using the points
    # generated above to determine the total sample‑detector distance.
    # Accumulate the result to obtain a random distribution of distances.
    # ######################################################################
    tic = time.perf_counter()
    Lsd_list = []                     # Lsd = []  in IDL

    for i in range(ns):               # for i=0,ns-1 do begin
        sx, sy, sz = SxCoords[i], SyCoords[i], SzCoords[i]

        for j in range(na):           # for j=0,na-1 do begin
            # distance sample → analyzer
            dx1 = AxCoords[j] - sx
            dy1 = AyCoords[j] - sy
            dz1 = AzCoords[j] - sz
            d1  = np.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)

            for k in range(nd):       # for k=0,nd-1 do begin
                # distance analyzer → detector
                dx2 = DxCoords[k] - AxCoords[j]
                dy2 = DyCoords[k] - AyCoords[j]
                dz2 = DzCoords[k] - AzCoords[j]
                d2  = np.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)

                Li = d1 + d2
                Lsd_list.append(Li)   # Lsd = [Lsd, Li]  in IDL

    Lsd = np.array(Lsd_list)          # turn list into NumPy array (1‑D)
    toc = time.perf_counter()
    print(f"{Lsd.size:,} distances evaluated in {toc - tic:.3f} s")

    # ######################################################################
    # Use statistical analysis to determine mean and standard deviation of
    # the distance and hence tof
    # ######################################################################
    mean_dist, var_dist = moment(Lsd)               # moments = Moment(Lsd) in IDL
    sigmaLambda = mean_dist * DeltaLambda / hOverMn

    fmt1 = "6.1f"
    fmt2 = "6.3f"

    # Build the same result strings that IDL prints
    results = [
        f"Wavelength = {wavelength:{fmt1}} Å",
        f"Mean S-D dist = {mean_dist*1000:{fmt1}} mm",
        f"$\\sigma$ of S-D dist = {np.sqrt(var_dist)*1000:{fmt1}} mm",
        f"Mean TOF      = {mean_dist*wavelength/hOverMn*1000:{fmt2}} ms",
        f"$\\sigma$ TOF         = {np.sqrt(var_dist)*wavelength/hOverMn*1000:{fmt2}} ms",
        f"$\\Delta t/t$    = {np.sqrt(var_dist)/mean_dist*100:4.2f} %",
        f"$\\sigma$ (due to $\\Delta\\lambda$)  = {sigmaLambda*1000:{fmt2}} ms",
    ]

    print("\n##################################################")
    print("*** CANDOR ***")
    print("##################################################")
    for line in results:
        print(line)
    print("##################################################")

    # ------------------------------------------------------------------
    # If the user asked for no plot, simply return here
    # ------------------------------------------------------------------
    if noPlotFlag:
        return Lsd, {"results": results, "moments": (mean_dist, var_dist),
                     "sigmaLambda": sigmaLambda}

    # ------------------------------------------------------------------
    # Histogram the results to enable a plot of the distribution of distances and times
    # ------------------------------------------------------------------
    frequency, distances = histogram(Lsd, nbins=51)    # frequency = Histogram(Lsd, nbins=51, locations=distances)
    tof = distances * wavelength / hOverMn

    # ---- Plot of distances ----------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(distances, frequency,
             marker='d', linestyle=' ', color='red')
    xtitle = r"L$_{samp-detector}$ (m)"
    ytitle = "Intensity"
    title  = f"CANDOR: S‑D Distances for $\\lambda$ = {wavelength:.2f} Å"
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)

    # Add the first three result strings (those that refer to distance)
    for i, txt in enumerate(results[:3]):
        plt.text(0.5, 0.7 - i*0.08, txt,
                 transform=plt.gca().transAxes,
                 fontfamily='Courier', fontsize=9,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    plt.tight_layout()
    plt.show()

    # ---- Plot of TOF ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(tof, frequency,
             marker='d', linestyle=' ', color='red')
    xtitle = r"TOF$_{samp-detector}$ (s)"
    title  = f"CANDOR: S‑D TOF for $\\lambda$ = {wavelength:.2f} Å"
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)

    # Add the remaining result strings (those that refer to TOF)
    for i, txt in enumerate(results[3:]):
        plt.text(0.5, 0.7 - i*0.08, txt,
                 transform=plt.gca().transAxes,
                 fontfamily='Courier', fontsize=9,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Return data that might be useful for other scripts
    # ------------------------------------------------------------------
    info = {
        "mean_distance": mean_dist,
        "variance_distance": var_dist,
        "sigma_lambda": sigmaLambda,
        "wavelength": wavelength,
        "DeltaLambda": DeltaLambda,
        "distances": distances,
        "frequency": frequency,
        "tof": tof,
        "results_strings": results,
    }

    return Lsd, info


# ----------------------------------------------------------------------
# Driver procedure for the TOF Estimate – translates `drive_TofEstForCANDOR`
# ----------------------------------------------------------------------
def drive_TofEstForCANDOR(detector_index: int = 0, nPoints: int = 0, noPlotFlag: bool = False) -> None:
    """
    Load the CANDOR parameter file (wavelength and HOPG/detector distances)
    from a text file, pick the detector row indicated by ``detector_index``,
    and then call :func:`ToFEstForCANDOR`.

    The original IDL code used ``sourcepath()+path_sep()`` to locate the file.
    Here we simply look for the file in the same directory as this script.
    """

    # ------------------------------------------------------------------
    # Retrieve CANDOR parameters file
    # ------------------------------------------------------------------
    script_dir = Path(__file__).parent
    par_file_name = "CANDOR_lambda_distances_bank0.txt"
    par_path = script_dir / par_file_name

    if not par_path.is_file():
        # In IDL a dialog is shown; in Python we raise an error.
        raise FileNotFoundError(f"Parameter file not found: {par_path}")

    # ------------------------------------------------------------------
    # Read CANDOR parameters from text file
    # The IDL code used:
    #   buffer = fltarr(6,nlines)
    #   Openr, lun, parfile, /get_lun
    #   Readf, lun, buffer
    #   Free_lun, lun, /force
    # Here we use NumPy's loadtxt which returns the same 6×N array.
    # ------------------------------------------------------------------
    buffer = np.loadtxt(par_path).T               # shape (6, nlines)
    nlines = buffer.shape[1]

    # Pre‑allocate the vectors that will hold each column (same order as IDL)
    lambda_arr   = np.empty(nlines)
    DeltaLambda  = np.empty(nlines)
    Lsa_arr      = np.empty(nlines)
    Lad_arr      = np.empty(nlines)
    Lsd_arr      = np.empty(nlines)

    for i in range(nlines):
        idx = int(buffer[0, i])                  # index = fix(buffer[0,i])
        lambda_arr[idx]   = buffer[1, i]
        DeltaLambda[idx]  = buffer[2, i]
        Lsa_arr[idx]      = buffer[3, i]
        Lad_arr[idx]      = buffer[4, i]
        Lsd_arr[idx]      = buffer[5, i]

    # ------------------------------------------------------------------
    # which detector index are we interested in evaluating?
    # detector_index = 0 > detector_index  ; must be >= 0
    # detector_index = detector_index < 53 ; must be <= 53
    # ------------------------------------------------------------------
    detector_index = max(0, detector_index)   # enforce >=0
    detector_index = min(53, detector_index)  # enforce <=53

    # ------------------------------------------------------------------
    # Call the Monte‑Carlo routine with the selected row
    # ------------------------------------------------------------------
    ToFEstForCANDOR(
        wavelength=lambda_arr[detector_index],
        DeltaLambda=DeltaLambda[detector_index],
        Lsa=Lsa_arr[detector_index],
        Lad=Lad_arr[detector_index],
        nPoints=nPoints,
    )


# ----------------------------------------------------------------------
# If this file is executed directly, run the driver with default args
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: evaluate detector 0 with the default 101 random points
    drive_TofEstForCANDOR(detector_index=0, nPoints=101)