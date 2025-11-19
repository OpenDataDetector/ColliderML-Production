#!/usr/bin/env python3
"""
Shared detector enum configuration for tracker and calorimeter.

This module defines:
- Integer codes for tracker and calorimeter detectors
- Helper functions to encode string detector names + geometry into enums

The goal is to keep all detector coding in one place so that:
- Converters remain simple and DRY
- Analysis code and documentation can reuse the same mapping
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


#: Tracker detector enum codes
TRACKER_DETECTOR_CODES: Dict[str, int] = {
    # Pixel
    "pixel_neg_endcap": 0,
    "pixel_barrel": 1,
    "pixel_pos_endcap": 2,
    # Short strips
    "short_neg_endcap": 3,
    "short_barrel": 4,
    "short_pos_endcap": 5,
    # Long strips
    "long_neg_endcap": 6,
    "long_barrel": 7,
    "long_pos_endcap": 8,
}


#: Calorimeter detector enum codes
CALO_DETECTOR_CODES: Dict[str, int] = {
    # Electromagnetic calorimeter
    "ecal_neg_endcap": 9,
    "ecal_barrel": 10,
    "ecal_pos_endcap": 11,
    # Hadronic calorimeter
    "hcal_neg_endcap": 12,
    "hcal_barrel": 13,
    "hcal_pos_endcap": 14,
}


def encode_tracker_detector(detector: pd.Series, z: pd.Series | None) -> pd.Series:
    """
    Encode tracker detector string names + z-sign into a uint8 enum.

    Inputs
    ------
    detector:
        Pandas Series of detector collection names (e.g. 'PixelBarrelReadout').
    z:
        Series giving the z or true_z coordinate for each row. Used to
        distinguish negative versus positive endcaps. If None, all endcaps
        are treated as positive.

    Returns
    -------
    encoded:
        Pandas Series of dtype uint8 with codes from TRACKER_DETECTOR_CODES.
        Unknown / unmatched detectors are encoded as 255.
    """
    det = detector.astype(str)
    if z is None:
        z_vals = pd.Series(0.0, index=det.index, dtype=float)
    else:
        z_vals = pd.to_numeric(z, errors="coerce").fillna(0.0)

    codes = np.full(len(det), 255, dtype="uint8")

    # Pixel
    pix_barrel = det == "PixelBarrelReadout"
    pix_endcap = det == "PixelEndcapReadout"
    codes[pix_barrel] = TRACKER_DETECTOR_CODES["pixel_barrel"]
    codes[pix_endcap & (z_vals < 0.0)] = TRACKER_DETECTOR_CODES["pixel_neg_endcap"]
    codes[pix_endcap & (z_vals >= 0.0)] = TRACKER_DETECTOR_CODES["pixel_pos_endcap"]

    # Short strips
    ss_barrel = det == "ShortStripBarrelReadout"
    ss_endcap = det == "ShortStripEndcapReadout"
    codes[ss_barrel] = TRACKER_DETECTOR_CODES["short_barrel"]
    codes[ss_endcap & (z_vals < 0.0)] = TRACKER_DETECTOR_CODES["short_neg_endcap"]
    codes[ss_endcap & (z_vals >= 0.0)] = TRACKER_DETECTOR_CODES["short_pos_endcap"]

    # Long strips
    ls_barrel = det == "LongStripBarrelReadout"
    ls_endcap = det == "LongStripEndcapReadout"
    codes[ls_barrel] = TRACKER_DETECTOR_CODES["long_barrel"]
    codes[ls_endcap & (z_vals < 0.0)] = TRACKER_DETECTOR_CODES["long_neg_endcap"]
    codes[ls_endcap & (z_vals >= 0.0)] = TRACKER_DETECTOR_CODES["long_pos_endcap"]

    return pd.Series(codes, index=det.index, dtype="uint8")


def encode_calo_detector(detector: pd.Series, z: pd.Series | None) -> pd.Series:
    """
    Encode calorimeter detector string names + z-sign into a uint8 enum.

    Inputs
    ------
    detector:
        Pandas Series of detector collection names
        (e.g. 'ECalBarrelCollection', 'HCalEndcapCollection').
    z:
        Series giving the z coordinate for each row. Used to distinguish
        negative versus positive endcaps. If None, all endcaps are treated
        as positive.

    Returns
    -------
    encoded:
        Pandas Series of dtype uint8 with codes from CALO_DETECTOR_CODES.
        Unknown / unmatched detectors are encoded as 255.
    """
    det = detector.astype(str)
    if z is None:
        z_vals = pd.Series(0.0, index=det.index, dtype=float)
    else:
        z_vals = pd.to_numeric(z, errors="coerce").fillna(0.0)

    codes = np.full(len(det), 255, dtype="uint8")

    # ECAL
    ecal_barrel = det == "ECalBarrelCollection"
    ecal_endcap = det == "ECalEndcapCollection"
    codes[ecal_barrel] = CALO_DETECTOR_CODES["ecal_barrel"]
    codes[ecal_endcap & (z_vals < 0.0)] = CALO_DETECTOR_CODES["ecal_neg_endcap"]
    codes[ecal_endcap & (z_vals >= 0.0)] = CALO_DETECTOR_CODES["ecal_pos_endcap"]

    # HCAL
    hcal_barrel = det == "HCalBarrelCollection"
    hcal_endcap = det == "HCalEndcapCollection"
    codes[hcal_barrel] = CALO_DETECTOR_CODES["hcal_barrel"]
    codes[hcal_endcap & (z_vals < 0.0)] = CALO_DETECTOR_CODES["hcal_neg_endcap"]
    codes[hcal_endcap & (z_vals >= 0.0)] = CALO_DETECTOR_CODES["hcal_pos_endcap"]

    return pd.Series(codes, index=det.index, dtype="uint8")



