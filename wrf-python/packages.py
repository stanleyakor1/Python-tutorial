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
from netCDF4 import Dataset
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
from matplotlib.colors import LogNorm
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

class precip_accumulate():
    def __init__(self, path, save_name, var_list):
        self.save_name = save_name
        self.var_list = var_list
        self.wrf = xr.open_dataset(path)
        self.wrf = self.wrf.swap_dims({'Time':'XTIME'})
        self.new_array = self.wrf[self.var_list].resample(XTIME = '24H').mean(dim = 'XTIME') # create daily means of few variables
        
    def calc_precip(cum_precip, bucket_precip):
        return  cum_precip + bucket_precip * 100.0

    # # Daily Aggregations (hourly to daily)
    def main(self):
        self.wrf['PRCP'] = self.wrf['RAINNC']
        self.wrf['PRCP'].values = self.calc_precip(self.wrf['RAINNC'],self.wrf['I_RAINNC'])
        
        # Compute accumulative precip for each day
        self.new_array['PRCP'] = self.wrf['PRCP'].resample(XTIME = '24H').max(dim = 'XTIME')
        
        new_array.to_netcdf(path)


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
        df = df[(44.35894 >= df['Latitude']) & (df['Latitude'] >= 42.603153) &\
                (-113.64972 >= df['Longitude']) & (df['Longitude'] >= -116.31619)]

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
        
        best = True
        worse = False
        
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
            wrf_precip = wrf_precip[2:108]
            bias = (df_filtered - wrf_precip).mean()
            self.feat[sta_id[id]] = bias
            names[sta_id[id]]=sta_names[id]

        # Get the keys with the smallest values (best 6)
        if best:
            smallest_keys = sorted(self.feat, key=lambda k: abs(self.feat[k]))[:6]

        # Get largest bias snotel guages (worse 6 guages)
        if worse:
            smallest_keys = sorted(self.feat, key=lambda k: abs(self.feat[k]))[-6:]
        # Create a new dictionary with the smallest keys and their corresponding values
        smallest_dict = {key:self.feat[key] for key in smallest_keys}
        
        # Basically extract the station names to make life easy in the next function
        # This definitely could be improved
        
        filtered_dict = {key: value for key, value in names.items()  if key in list(smallest_dict.keys())}
        # #print(filtered_dict)
        return smallest_dict, filtered_dict

    def read_csv2(self):
        
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
            wrf_precip = wrf_precip[2:108]
            bias = (df_filtered - wrf_precip).mean()
            self.feat[sta_id[id]] = bias
            names[sta_id[id]]=sta_names[id]

        
        smallest_keys = sorted(self.feat, key=lambda k: abs(self.feat[k]))
       
        # Create a new dictionary with the smallest keys and their corresponding values
        smallest_dict = {key:self.feat[key] for key in smallest_keys}
        
        # # Basically extract the station names to make life easy in the next function
        # # This definitely could be improved
        
        filtered_dict = {key: value for key, value in names.items()  if key in list(smallest_dict.keys())}
        # # #print(filtered_dict)
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
            wrf_precip = self.extract_precip(ixlat,ixlon)[2:108] # should be adjust according to the input data
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


'''
Compare snotel values across multiple WRF microphysics schemes.
ToDos:
(1) Tidy functions
(2) Very results
(3) Clean the code
(4) Add option if the smallest bias sites aren't same across schemes
(5) Unify the dates in the nc files

'''
class CompareScheme(getclosest):  
    def __init__(self, path_to_header, path_to_csv, path_to_geog, path_to_wrf_file, save_name, reference, save = True):
        super().__init__(path_to_header, path_to_csv, path_to_geog, path_to_wrf_file, save_name,save)
        self.allfile = {}
        self.check = False

        '''
        Initialize the first file, this certainly should be improved!
        
        '''
        self.allfile[reference] = self.wrf
    def compare_multiple(self,diction):
        for key,value in diction.items():
            self.wrf = xr.open_dataset(value)
            self.allfile[key] = self.wrf
        return self.allfile

    # read multiple wrf files (schemes)
    def read_multiple(self,diction):
        all_file = self.compare_multiple(diction)
        all_files_dict = {}
        for key , file in all_file.items():
            self.wrf_file = file['PRCP']
            smallest_dict, filtered_dict = self.read_csv()
            all_files_dict[key] = smallest_dict,filtered_dict

           
        # Get the keys from the first sub-dictionary
        keys_to_check = set(all_files_dict[next(iter(all_files_dict))][0].keys())
        #print(keys_to_check)
        
        # Iterate through the rest of the sub-dictionaries
        for key in all_files_dict:
            sub_dict_keys = all_files_dict[key][0].keys()
            if keys_to_check != set(sub_dict_keys):
                print(f"Keys in sub-dictionary '{key}' do not match.")
                break
        else:
            print("Keys in all sub-dictionaries match.")


        # If all schemes have best estimate at the same time, proceed
        #if self.check:
        return all_files_dict

    def extract(self,file,ixlat,ixlon):
        return file.isel(south_north = ixlat, west_east = ixlon).values
        
    def smallest(self,diction):
        all_dict = self.get_wrf_xy()
        allfiles = self.read_multiple(diction)
        schemes = list(allfiles.keys())
        wrf_files = self.compare_multiple(diction)

        # Since all the sites are the same for all schemes
        # Extract sites ID for only one of them
        filtered_dict = next(iter(allfiles.items()))[1][1]
        filtered_dict = dict(sorted(filtered_dict.items()))
        allkeys = np.array(list(filtered_dict.keys()))
        
        #print(filtered_dict)
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
            #print(ixlat,ixlon)
            ax.plot(date_range,df_filtered,'r--', label=f'snotel')

            for key1,value in wrf_files.items():
                
                wrf_precip = self.extract(value['PRCP'],ixlat,ixlon)[2:108] # should be adjust according to the input data
                ax.plot(date_range,wrf_precip, label=f'{key1}')

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

        # if self.save:
        #     plt.savefig(self.save_name+'.pdf',dpi=600)
            
        plt.show()
'''
    Make spatial plot of a variable, and represent the bias in snotel estimates with a colour on the grid.

'''

class plot_snotel_grid(CompareScheme):
    def __init__(self, path_to_header, path_to_csv, path_to_geog, path_to_wrf_file,\
                 save_name, reference, save = True):
        super().__init__(path_to_header, path_to_csv, path_to_geog, path_to_wrf_file, save_name,reference,save)
        self.wrf_data = xr.open_dataset(path_to_geog)
        self.elevation = self.wrf_data['HGT'].isel(Time=0)
        self.lat,self.lon = self.wrf_data['XLAT'].isel(Time=0),self.wrf_data['XLONG'].isel(Time=0)
        self.path = path_to_geog
                     
    def multiple_points(self):
        a,b = self.read_csv2()
        lat, lon, sta_names, sta_id = self.collect_snodas_info()
        fig, ax = plt.subplots(figsize=(12, 12))
        for key, value in a.items():
            index = sta_id.index(key)
            la, lo = lat[index], lon[index]
        
            if abs(value) >= 0 and abs(value) <= 10:
                ax.scatter(lo, la, marker='D', s=15, color='black')
                ax.text(lo - 0.02, la + 0.005, key, size=9, weight='bold')
        
            elif abs(value) >= 11 and abs(value) <= 20:
                ax.scatter(lo, la, marker='D', s=15, color='red')
                ax.text(lo - 0.02, la + 0.005, key, size=9, weight='bold')
        
            elif abs(value) >= 21 and abs(value) <= 30:
                ax.scatter(lo, la, marker='D', s=15, color='blue')
                ax.text(lo - 0.02, la + 0.005, key, size=9, weight='bold')
        
            else:
                ax.scatter(lo, la, marker='D', s=15, color='green')
                ax.text(lo - 0.02, la + 0.005, key, size=9, weight='bold')
        
        im = ax.imshow(self.elevation, extent=(self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()), cmap='terrain', origin='lower', alpha=1.0)

  
        plt.title('Elevation (m)')

        # Modify latitude and longitude labels
        lon_ticks = ax.get_xticks()
        lat_ticks = ax.get_yticks()
        lon_labels = [f'{abs(self.lon):.2f}°{"W" if self.lon < 0 else "E"}' for self.lon in lon_ticks]
        lat_labels = [f'{abs(self.lat):.2f}°{"S" if self.lat < 0 else "N"}' for self.lat in lat_ticks]
        ax.set_xticklabels(lon_labels)
        ax.set_yticklabels(lat_labels)
        
        # Add gridlines
        ax.grid(color='gray', linestyle='--', linewidth=0.0001)
        
        # Create a colorbar with the same height as the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.show()