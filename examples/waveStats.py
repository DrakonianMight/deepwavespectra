"""
Wave Parameters Calculator

This module provides functions to calculate various wave parameters from the wave spectrum.

Author: lpeach
Date: 01-06-2024
"""

# Import necessary libraries here
import pandas as pd
import numpy as np
import xarray as xr


R2D = 180.0 / np.pi

def Hm0(wave_spectrum: xr.DataArray) -> float:
    # Ensure the frequencies and spectrum arrays are numpy arrays
    frequency_spectrum = wave_spectrum.integrate('dir')
    frequencies = frequency_spectrum.freq.values
    # Calculate the zero-th moment (m0) using the trapezoidal rule
    m0 = np.trapz(frequency_spectrum, frequencies)
    
    # Calculate Hm0
    Hm0 = 4 * np.sqrt(m0)
    
    return np.round(Hm0.item(), 2)

def Hs(spectrum, freq, dir=None, tail=True):
    """Significant wave height Hmo.

    Args:
        - spectrum (2darray): wave spectrum array.
        - freq (1darray): wave frequency array.
        - dir (1darray): wave direction array.
        - tail (bool): if True fit high-frequency tail before integrating spectra.

    Returns:
        - hs (float): Significant wave height.

    """
    df = abs(freq[1:] - freq[:-1])
    if dir is not None and len(dir) > 1:
        ddir = abs(dir[1] - dir[0])
        E = ddir * spectrum.sum(1)
    else:
        E = np.squeeze(spectrum)
    Etot = 0.5 * sum(df * (E[1:] + E[:-1]))
    if tail and freq[-1] > 0.333:
        Etot += 0.25 * E[-1] * freq[-1]
    return 4.0 * np.sqrt(Etot)

def Tp(wave_spectrum: xr.DataArray) -> float:
    """
    Calculate the peak period (Tp) from a directional wave spectrum.

    Parameters:
    wave_spectrum (xarray.DataArray): The wave spectrum, assumed to be a 2D array with dimensions 'frequency' and 'direction'.

    Returns:
    float: The peak period (Tp).
    """
    # Calculate the frequency spectrum by integrating over direction
    frequency_spectrum = wave_spectrum.sum('dir')

    # Find the frequency at which the spectrum is maximum
    f_peak = frequency_spectrum.where(frequency_spectrum == frequency_spectrum.max(), drop=True)['freq'].values[0]

    # Calculate the peak period
    Tp = 1 / f_peak

    return Tp.round(2)

def Dp(wave_spectrum: xr.DataArray) -> float:
    """
    Calculate the peak wave direction from a directional wave spectrum.

    Parameters:
    wave_spectrum (xarray.DataArray): The wave spectrum, assumed to be a 2D array with dimensions 'frequency' and 'direction'.

    Returns:
    float: The peak wave direction.
    """
    # Calculate the frequency spectrum by integrating over direction
    frequency_spectrum = wave_spectrum.sum('dir')

    # Find the frequency at which the spectrum is maximum
    f_peak = frequency_spectrum.where(frequency_spectrum == frequency_spectrum.max(), drop=True)['freq'].values[0]

    # Get the directional spectrum at the peak frequency
    directional_spectrum = wave_spectrum.sel(freq=f_peak)

    # Find the direction at which the directional spectrum is maximum
    peak_direction = directional_spectrum.where(directional_spectrum == directional_spectrum.max(), drop=True)['dir'].values[0]

    return peak_direction


def Tm02(wave_spectrum: xr.DataArray) -> float:
    """
    Calculate the zero-upcrossing period (Tm02) from a directional wave spectrum.

    Parameters:
    wave_spectrum (xarray.DataArray): The wave spectrum, assumed to be a 2D array with dimensions 'frequency' and 'direction'.

    Returns:
    float: The zero-upcrossing period (Tm02).
    """
    # Calculate the frequency spectrum by integrating over direction
    frequency_spectrum = wave_spectrum.sum('dir')

    # Calculate the zeroth moment of the frequency spectrum
    m0 = frequency_spectrum.sum('freq')

    # Calculate the second moment of the frequency spectrum
    m2 = (frequency_spectrum * wave_spectrum['freq']**2).sum('freq')

    # Calculate the zero-upcrossing period
    Tm02 = np.sqrt(m0 / m2)

    return np.round(Tm02.item(), 2)

def mom1(spectrum, dir, theta=90.0):
    """First directional moment.

    Args:
        - spectrum (2darray): wave spectrum array.
        - dir (1darray): wave direction array.
        - theta (float): angle offset.

    Returns:
        - msin (float): Sin component of the 1st directional moment.
        - mcos (float): Cosine component of the 1st directional moment.

    """
    dd = dir[1] - dir[0]
    cp = np.cos(np.radians(180 + theta - dir))
    sp = np.sin(np.radians(180 + theta - dir))
    msin = (dd * spectrum * sp).sum(axis=1)
    mcos = (dd * spectrum * cp).sum(axis=1)
    return msin, mcos


def Dm(spectrum, dir):
    """Mean wave direction Dm.

    Args:
        - spectrum (2darray): wave spectrum array.
        - dir (1darray): wave direction array.

    Returns:
        - dm (float): Mean spectral period.

    """
    moms, momc = mom1(spectrum, dir)
    dm = np.arctan2(moms.sum(axis=0), momc.sum(axis=0))
    dm = (270 - R2D * dm) % 360.0
    return dm

def calculate_wave_parameters(wave_spectrum: xr.DataArray) -> pd.DataFrame:
    """
    Calculate various wave parameters from a directional wave spectrum.

    Parameters:
    wave_spectrum (xarray.DataArray): The wave spectrum, assumed to be a 2D array with dimensions 'frequency' and 'direction'.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated wave parameters.
    """
    # Calculate wave parameters
    hm0 = Hs(wave_spectrum.values, wave_spectrum['freq'].values, wave_spectrum['dir'].values)
    tp = Tp(wave_spectrum)
    dm = Dm(wave_spectrum.values, wave_spectrum['dir'].values)
    dp = Dp(wave_spectrum)
    tm02 = Tm02(wave_spectrum)
    
    # Create DataFrame with wave parameters
    wave_parameters = pd.DataFrame({
        'Hm0': [hm0],
        'Tp': [tp],
        'Dm': [dm],
        'Peak Direction': [dp],
        'Tm02': [tm02],
        'Timestep': [wave_spectrum.time.values]
    })
    # Set the time variable as the index of the DataFrame
    wave_parameters.index = wave_parameters.Timestep
    return wave_parameters