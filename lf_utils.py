from kcorr import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import math
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import LambdaCDM
from astropy . coordinates import Distance
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from tqdm.notebook import tqdm

def sample(dataframe, n, reset_index=True, random_state=None):
    """
    Returns a sample of an input dataframe.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe.
    n (int): Number of rows to sample.
    reset_index (bool): Whether to reset the index of the returned sample.
    random_state (int, optional): Seed for reproducibility.
    
    Returns:
    pd.DataFrame: Sampled dataframe.
    """
    
    if n > len(dataframe):
        raise ValueError("Sample size n cannot be greater than the number of rows in the dataframe")
    
    sample_dataframe = dataframe.sample(n, random_state=random_state)
    
    if reset_index:
        sample_dataframe = sample_dataframe.reset_index(drop=True)
    
    return sample_dataframe

def add_column(dataframe, column_file='DistancesFramesv14.fits', column_name='Z_TONRY', common_column='CATAID'):
    """
    Adds a column from a fits file to an input dataframe.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe.
    column_file (str): Fits file to extract the column from.
    column_name (str): Name of the column to extract.
    common_column (str): Name of the common column between the dataframe and the column_file in order to merge onto the dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with the extra column.
    """
    if isinstance(column_name, str):
        column_name = [column_name]
    
    hdul = fits.open(column_file)
    data = hdul[1].data
    t = Table(data)
    column_dataframe = t.to_pandas()    

    selected_columns = [common_column] + column_name
    merged_dataframe = pd.merge(dataframe, column_dataframe[selected_columns], on=common_column, how='left', suffixes=('', '_new'))

    for col in column_name:
        new_col = col + '_new'
        if new_col in merged_dataframe.columns:
            merged_dataframe[col] = merged_dataframe[new_col].combine_first(merged_dataframe[col])
            merged_dataframe.drop(columns=[new_col], inplace=True)

    return merged_dataframe

def kcorrection(dataframe, zrange = [0.002, 0.65], z0 = 0, pdeg = 4, ntest = 0, 
                responses = ['galex_FUV', 'galex_NUV', 'vst_u', 'vst_g', 'vst_r', 'vst_i', 'vista_z', 'vista_y', 'vista_j', 'vista_h', 'vista_k', 'wise_w1', 'wise_w2'], 
                fnames = ['flux_FUVt', 'flux_NUVt', 'flux_ut', 'flux_gt', 'flux_rt', 'flux_it', 'flux_Zt', 'flux_Yt', 'flux_Jt', 'flux_Ht', 'flux_Kt', 'flux_W1t', 'flux_W2t'], 
                ferrnames = ['flux_err_FUVt', 'flux_err_NUVt', 'flux_err_ut', 'flux_err_gt', 'flux_err_rt', 'flux_err_it', 'flux_err_Zt', 'flux_err_Yt', 'flux_err_Jt', 'flux_err_Ht', 'flux_err_Kt', 'flux_err_W1t', 'flux_err_W2t'], 
                ref_index = 4, avg_index = 6, redshift = 'Z', survey='GAMAIII'):
    """performs k-corrections on the data in the input dataframe. Returned dataframe contains k-correction and pcoeffs columns"""
    
    kcorrect_dataframe = kcorr_gkv(dataframe, zrange, z0, pdeg, ntest, responses, fnames, ferrnames, ref_index, avg_index, redshift)
    return kcorrect_dataframe

def luminosity_distance(dataframe, redshift='Z', H0=100, Om0=0.3, Ode0=0.7):
    """calculates the luminosity distance for the data in the input dataframe"""
    
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    dataframe['Lum_Distance'] = cosmo.luminosity_distance(dataframe[redshift].values).to(u.parsec).value
    return dataframe

def magnitude(dataframe, 
              bands = ['FUV', 'NUV', 'u', 'g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'K', 'W1', 'W2'], 
              fluxbands = ['flux_FUVt', 'flux_NUVt', 'flux_ut', 'flux_gt', 'flux_rt', 'flux_it', 'flux_Zt', 'flux_Yt', 'flux_Jt', 'flux_Ht', 'flux_Kt', 'flux_W1t', 'flux_W2t'], 
              lumdist = 'Lum_Distance', kcorrection = 'Kcorrection'):
    """calculates the apparent and absolute magnitudes of the data in the input dataframe"""
    
    for i in range(len(fluxbands)):
        dataframe[f'm_{bands[i]}'] = 8.9 - 2.5 * np.log10(dataframe[fluxbands[i]])
        dataframe[f'M_{bands[i]}'] = dataframe[f'm_{bands[i]}'] - 5 * np.log10(dataframe[lumdist]) + 5 - [x[i] for x in dataframe[kcorrection]]
    return dataframe

def colour(dataframe, 
           mag = ['m_FUV', 'm_NUV', 'm_u', 'm_g', 'm_r', 'm_i', 'm_Z', 'm_Y', 'm_J', 'm_H', 'm_K', 'm_W1', 'm_W2'], 
           bands = ['FUV', 'NUV', 'u', 'g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'K', 'W1', 'W2'], 
           ref_mag = 'm_r'):
    """calculates the colour of adjacent bands and of bands with respect to the r band"""
    
    for i in range(len(mag)-1):
        dataframe[f'{bands[i]}-{bands[i+1]}_colour'] = dataframe[mag[i]] - dataframe[mag[i+1]]
        
    index = mag.index(ref_mag)
    for i in range(len(mag)):
        if i < index:
            dataframe[f'{bands[i]}-{bands[index]}_colour'] = dataframe[mag[i]] - dataframe[mag[index]]
        if i == index:
            dataframe[f'{bands[index]}-{bands[index]}_colour'] = dataframe[mag[index]] - dataframe[mag[index]]
        if i > index:
            dataframe[f'{bands[index]}-{bands[i]}_colour'] = dataframe[mag[index]] - dataframe[mag[i]]
    
    return dataframe
