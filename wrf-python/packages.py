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

from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross, ll_to_xy,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair, destagger, 
                 interplevel)

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

        da_rain_acc = xr.concat([temp2,temp3], 'XTIME')

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


class getclosest():
    def __init__(self, path_to_header,path_to_csv,path_to_geog,path_to_wrf_file,save_name, save = True):
        self.path_to_header = path_to_header
        self.path_to_csv = path_to_csv
        self.geog = xr.open_dataset(path_to_geog)
        self.dict = {}
        self.wrf = xr.open_dataset(path_to_wrf_file)
        self.wrf_file = self.wrf['PRCP']
        self.start = '2021-10-01'
        self.end = '2022-01-14'
        self.feat = {}
        self.save = save
        self.save_name = save_name
        #self.wrf_path = path_to_wrf_file

    def collect_snodas_info(self):
        df = pd.read_csv(self.path_to_header)
        df = df[(df['Latitude'] >= 42.80) & (df['Longitude'] > -116)]
        filtered_df = df[df['State'] == 'ID']
        sta_names = filtered_df['Station Name'].tolist()
        lat = filtered_df['Latitude'].tolist()
        lon = filtered_df['Longitude'].tolist()
        sta_id = filtered_df['Station ID'].tolist()
        #print(sta_id)

        '''
          Seems one of the stations has been 
          decommissioned, so we remove it

        '''
        # element_to_find = 782
        # index = sta_id.index(element_to_find)
        # del lat[index]
        # del lon[index]
        # del sta_names[index]
        # del sta_id[index]
        return lat,lon,sta_names,sta_id

    def get_wrf_xy(self):
        xlat = self.geog.XLAT.values[0,:,:]
        xlon = self.geog.XLONG.values[0,:,:]

        lat, lon, sta_names, sta_id = self.collect_snodas_info()
        
        for x,y,z in zip(lat,lon,sta_id):
            dist = np.sqrt((xlat - x)**2 + (xlon - y)**2)
            mindist = dist.min()
            ixlat = np.argwhere(dist == mindist)[0][0]
            ixlon = np.argwhere(dist == mindist)[0][1]

            self.dict[str(z)] = (ixlat,ixlon)
            
        return self.dict

    def extract_precip(self,ixlat,ixlon):
        return self.wrf_file.isel(south_north = ixlat, west_east = ixlon).values
    
    def read_csv(self):
    
        lat, lon, sta_names, sta_id = self.collect_snodas_info()
        dict = self.get_wrf_xy()
        names = {}
        for id in range(len(sta_id)):
            name = f'df_{sta_id[id]}.csv'
            path = self.path_to_csv+'/'+name
            generic = f'{sta_names[id]} ({sta_id[id]}) Precipitation Accumulation (in) Start of Day Values'
            df = pd.read_csv(path)
            df = df[(df['Date'] >=self.start) & (df['Date'] <= self.end)]
            df_filtered = df[generic].tolist()
            df_filtered = [value * 25.4 for value in df_filtered]
            ixlat,ixlon = dict[str(sta_id[id])]
            wrf_precip = self.extract_precip(ixlat,ixlon)
            wrf_precip = wrf_precip[2:-1]
            bias = (df_filtered - wrf_precip).mean()
            self.feat[sta_id[id]] = bias
            names[sta_id[id]]=sta_names[id]

        # Get the keys with the 5 smallest values
        smallest_keys = sorted(self.feat, key=lambda k: abs(self.feat[k]))[:6]

        # Create a new dictionary with the smallest keys and their corresponding values
        smallest_dict = {key:self.feat[key] for key in smallest_keys}
        
        # Basically extract the station names to make life easy in the next function
        # This definitely could be improved
        
        filtered_dict = {key: value for key, value in names.items()  if key in list(smallest_dict.keys())}
        # #print(filtered_dict)
        return smallest_dict, filtered_dict

    def compare_smallest(self):
        all_dict = self.get_wrf_xy()
        dict, filtered_dict = self.read_csv()
        allkeys = np.array(list(dict.keys()))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
        for index, key in enumerate(allkeys):
            ax = axes[index // 3, index % 3]
            row, col = divmod(index, 3)
            name = f'df_{key}.csv'
            path = self.path_to_csv+'/'+name
            generic = f'{filtered_dict[key]} ({key}) Precipitation Accumulation (in) Start of Day Values'
            df = pd.read_csv(path)
            df = df[(df['Date'] >=self.start) & (df['Date'] <= self.end)]
            date_range = pd.date_range(self.start, self.end, freq='1D')
            df_filtered = df[generic].tolist()
            df_filtered = [value * 25.4 for value in df_filtered]
            ixlat,ixlon = all_dict[str(key)]
            wrf_precip = self.extract_precip(ixlat,ixlon)[2:-1] # should be adjust according to the input data
            ax.plot(date_range,df_filtered,'r--', label=f'snotel')
            ax.plot(date_range,wrf_precip, label=f'WSM6')
            ax.set_title(f'{filtered_dict[key]} ({key})')
            ax.legend()
            
            if row == 1:  # Only for the bottom row
                ax.set_xlabel('Day')
            else:
                ax.set_xlabel('')
                
            if col == 0:
                ax.set_ylabel('Precipitation (mm)')
            ax.legend()
        
                    # Set x-axis locator and formatter for exterior plots
            if row ==1:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.xaxis.set_major_locator(plt.NullLocator())  # Hide x-axis locator
                ax.set_xlabel('')  # Set x-label to empty string
            
        # Adjust layout and show plots
        plt.tight_layout()

        if self.save:
            plt.savefig(self.save_name+'.pdf',dpi=600)
            
        plt.show()