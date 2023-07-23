
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

#subset wrfout data following  CHALRIE BECKER, 2019.


def calc_precip(cum_precip, bucket_precip):
    
    total_precip = cum_precip + bucket_precip * 100.0
    PRCP = np.zeros(total_precip.shape)
    
    for i in np.arange(1,PRCP.shape[0]):
        
        PRCP[i,:,:] = total_precip[i,:,:].values - total_precip[i-1,:,:].values
        
    return PRCP

if __name__ == '__main__':
	scheme = 'thompson'
	path = '/bsuhome/stanleyakor/scratch/'+ scheme +'/' 

	home_dir = os.path.expanduser("~")
	output_dir = os.path.join(home_dir,'DJF',scheme,'/')

	files =  sorted(glob.glob(path + 'wrfout_d02_2022-12-01*'))
	d = xr.open_mfdataset(files, concat_dim='Time',combine='nested')

	d = d.swap_dims({'Time':'XTIME'})

	# Daily Aggregations (hourly to daily)
	d['PRCP'] = d['RAINNC']
	d['PRCP'].values = calc_precip(d['RAINNC'],d['I_RAINNC'])
	new_array = d[['SWDOWN','SWNORM','Q2','T2','SNOWH']].resample(XTIME = '24H').mean(dim = 'XTIME') # create daily means of few variables
	new_array['TMIN'] = d['T2'].resample(XTIME = '24H').min(dim = 'XTIME') # create daily minimum temperature
	new_array['TMAX'] = d['T2'].resample(XTIME = '24H').max(dim = 'XTIME')  # create daily maximum temperature
	new_array['PRCP'] = d['PRCP'].resample(XTIME = '24H').sum(dim = 'XTIME')

    # rename T2 as TMEAN
	new_array = new_array.rename({'T2' : 'TMEAN'}) 

    # Adjust some meta data
	new_array['TMEAN'].attrs = [('description','DAILY MEAN GRID SCALE TEMPERATUTE'), ('units','K')]
	new_array['TMIN'].attrs = [('description','DAILY MINIMUM GRID SCALE TEMPERATURE'), ('units','K')]
	new_array['TMAX'].attrs = [('description','DAILY MAXIMUM GRID SCALE TEMPERATURE'), ('units','K')]
	new_array['Q2'].attrs = [('description','DAILY MEAN GRID SCALE SPECIFIC HUMIDITY'), ('units','')]
	new_array['SWDOWN'].attrs = [('description','DAILY MEAN DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE'), ('units','W m^2')]
	new_array['SWNORM'].attrs = [('description','DAILY MEAN NORMAL SHORT WAVE FLUX AT GROUND SURFACE (SLOPE-DEPENDENT)'), ('units','W m^2')]
	new_array['SNOWH'].attrs = [('description','DAILY Mean Snow Height'), ('units','m')]
	new_array['PRCP'].attrs = [('description','DAILY ACCUMULATED GRID SCALE PRECIPITATION'), ('units','mm')]
    
    # Write new netcdf file
	if not os.path.exists(output_dir):
		 try:
		 	os.makedirs(output_dir)
		 	new_array.to_netcdf(output_dir + scheme + '_DJF.nc')
		 	print("Successfully wrote file")

		 except OSError as e:
	        
		 	print(f"Error creating the directory: {e}")
	
	new_array.to_netcdf(output_dir + scheme + '_DJF'+".nc")
	
	del d, new_array
