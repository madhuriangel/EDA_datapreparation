#Merged all the nc file of the noaa data
import xarray as xr
import glob

# Define the bounding box region
lon_min, lon_max = 347, 356
lat_min, lat_max = 49, 56

# Get a list of all NetCDF files in the specified directory
file_list = glob.glob('Data_noaa_copernicusnoaa_avhrr/*.nc')

# Open and concatenate the files along the time dimension
ds = xr.open_mfdataset(file_list, combine='by_coords')

# Select the desired region using the latitude and longitude bounds
ds_region = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

# Remove the zlev dimension
ds_region = ds_region.squeeze(dim='zlev')

# Save the merged dataset to a new NetCDF file
ds_region.to_netcdf('Data_noaa_copernicus/noaa_avhrr/merged_data.nc')
