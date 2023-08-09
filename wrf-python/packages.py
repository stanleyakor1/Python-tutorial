import xarray as xr
import xesmf as xe
import numpy as np 
import pathlib as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import glob
import subprocess
import os
import multiprocessing as mp
import sys
import shutil
import warnings
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import ImageGrid
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Block all future warnings
warnings.filterwarnings("ignore")


def hourly_to_daily(input_path, input_file, in_var_name, out_var_name, operator, description, units, rain_bucket_vol):
    
    assert (operator=='sum' or operator=='mean' or operator=='max' or operator=='min'), \
        "Input operator must be sum, mean, min, or max" 

    # Open dataset, error check
    ds_wrf = xr.open_dataset(input_path+input_file)

    assert ds_wrf, "Could not open dataset with "+input_path+input_file

    ds_wrf = ds_wrf.swap_dims({'Time': 'XTIME'})

    # Special case: if one of the precipitation variables is RAINNC or RAINC, then
    # I_RAINNC or I_RAINC, respectively, needs to be added in before the resampling
    # makes any sense
    if((in_var_name=='RAINNC') or (in_var_name=='RAINC')):
        da_rain  = ds_wrf[in_var_name]
        da_irain = ds_wrf['I_'+in_var_name]

        temp1 = rain_bucket_vol*da_irain + da_rain
        temp2 = 0.0*temp1.isel(XTIME=0)
        temp3 = temp1.diff('XTIME')

        da_rain_acc = xr.concat([temp2, temp3], 'XTIME')

        in_var_name = in_var_name+'_ACC'

        ds_wrf[in_var_name] = da_rain_acc


    if (operator=='sum'):
        da_wrf = ds_wrf[in_var_name].resample(XTIME='1D').sum(dim='XTIME')
    elif (operator=='mean'):
        da_wrf = ds_wrf[in_var_name].resample(XTIME='1D').mean(dim='XTIME')
    elif (operator=='min'):
        da_wrf = ds_wrf[in_var_name].resample(XTIME='1D').min(dim='XTIME')
    elif (operator=='max'):
        da_wrf = ds_wrf[in_var_name].resample(XTIME='1D').max(dim='XTIME')

    ds_wrf_new = da_wrf.to_dataset(name=out_var_name)

    if(in_var_name!='XTIME'):
        ds_wrf_new[out_var_name].attrs = [('description', description),('units',units)]

    return ds_wrf_new

def max_swe(ds):
    day = 0
    max = 0
    for i in range(ds.shape[0]):
        if(ds.values.max() > max):
            max = ds.values.max()
            day = i

    return i

def get_wrf_xy(geog,lat, lon):
    #lat = 38.89
    #lon = -106.95
    xlat = geog.XLAT.values[0,:,:]
    xlon = geog.XLONG.values[0,:,:]
    dist = np.sqrt((xlat - lat)**2 + (xlon - lon)**2)
    mindist = dist.min()
    ixlat = np.argwhere(dist == mindist)[0][0]
    ixlon = np.argwhere(dist == mindist)[0][1]
    return ixlat, ixlon

def make_snodas_Wrf_plots(file_list, title, lat, lon, save_title,label, show=True, subplots=(1,3), colour='terrain', save=True):

    vmax = max(np.nanmax(ds.values) for ds in file_list)
    vmin = min(np.nanmin(ds.values) for ds in file_list)

    fig = plt.figure(figsize=(10, 15))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=subplots,
                 axes_pad=0.3,
                 share_all=True,
                 cbar_location="bottom",
                 cbar_mode="single",
                 cbar_size="4%",  
                 cbar_pad=0.15,
                )

    for i, ax in enumerate(grid):
        im = ax.imshow(file_list[i],extent=(lon.min(), lon.max(), lat.min(), lat.max()),vmax=vmax, vmin = vmin, cmap=colour, origin='lower', alpha=1.0)
        ax.set_title(title[i])
        ax.xaxis.set_major_locator(plt.MultipleLocator(base=1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(base=0.4))
   

    # for ax in ax.flat:
    lon_ticks = ax.get_xticks()
    lat_ticks = ax.get_yticks()
    lon_labels = [f'{abs(lon):.2f}°{"W" if lon < 0 else "E"}' for lon in lon_ticks]
    lat_labels = [f'{abs(lat):.2f}°{"S" if lat < 0 else "N"}' for lat in lat_ticks]
    ax.set_xticklabels(lon_labels)
    ax.set_yticklabels(lat_labels)
    
    # Colorbar
    cbar = ax.cax.colorbar(im)
    cbar_X = ax.cax.toggle_label(True)
    cbar.set_label(label)

    if save:
        plt.savefig(save_title+'.pdf',dpi=600)
    
    if show:
        plt.show()