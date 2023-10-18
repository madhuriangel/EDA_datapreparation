#This is the final noaa sst data,
#Changed the datetime, to just date
import xarray as xr
import pandas as pd

# Open the existing NetCDF file
data = xr.open_dataset('Data_collection/noaa_avhrr/merged_data_sst.nc')

# Create a new array of dates from 1982-01-01 to 2022-12-31
new_dates = pd.date_range(start='1982-01-01', end='2022-12-31', freq='D')

# Assign the new array of dates to the 'time' dimension
data['time'] = new_dates

# Save the modified data as a new NetCDF file
data.to_netcdf('Data_collection/noaa_avhrr/final_noaasst_data.nc')
