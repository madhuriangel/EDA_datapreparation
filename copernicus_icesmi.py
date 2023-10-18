import xarray as xr
import numpy as np

"""
This part of the code converts the Kelvin unit to Celsius of the Copernicus data

############################################################################################

def kelvin_to_celsius(input_file, output_file):
    # Load the input netCDF file
    data = xr.open_dataset(input_file)

    # Convert 'analysed_sst' from Kelvin to Celsius
    data['analysed_sst'] = data['analysed_sst'] - 273.15

    # Save the modified data as a new netCDF file
    data.to_netcdf(output_file)

# Call the function with your input and output file paths
input_file = 'Data_collection/Copernicus/cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE_1691196251835.nc'
output_file = 'Data_collection/Copernicus/Copernicus_sst_Celsius.nc'

kelvin_to_celsius(input_file, output_file)

"""
def combine_and_fill_nan(input_file1, input_file2, output_file):
    # Load the input netCDF files
    data1 = xr.open_dataset(input_file1)
    data2 = xr.open_dataset(input_file2)

    # Replace NaN values in data1 with corresponding values from data2
    data1['analysed_sst'] = xr.DataArray(
        np.where(np.isnan(data1['analysed_sst']), data2['analysed_sst'], data1['analysed_sst']),
        dims=data1['analysed_sst'].dims,
        coords=data1['analysed_sst'].coords,
        name='analysed_sst'
    )
    
    # Add metadata attributes to the variables
    data1['analysed_sst'].attrs['long_name'] = 'Analyzed Sea Surface Temperature'
    data1['analysed_sst'].attrs['units'] = 'Celsius'

    # Add global metadata attributes to the dataset
    data1.attrs['title'] = 'Combined Sea Surface Temperature Data from ICES, MI and Copernicus'
    data1.attrs['Source1'] = 'Marine Institute (MI)'
    data1.attrs['Source2'] = 'International Council for the Exploration of the Seas (ICES)'
    data1.attrs['Source3'] = 'Copernicus'
    data1.attrs['University'] = 'Atlantic Technological University'
    data1.attrs['Department'] = 'Marine and Freshwater Research Centre'
    data1.attrs['history'] = 'Created using custom Python script'
    data1.attrs['Conventions'] = 'CF-1.8'
    data1.attrs['Temporal Resolution']  = 'Daily from 1/1/1982 to 31/12/2020'
    data1.attrs['Spatial Resolution'] = 'Gridded 0.05*0.05'
    # Save the modified data as a new netCDF file
    data1.to_netcdf(output_file)

# Call the function with your input and output file paths
input_file1 = 'Data_noaa_copernicus/Copernicus/copernicus_icesmi_sst0.05data.nc'
input_file2 = 'Data_noaa_copernicus/Copernicus/Copernicus_sst_Celsius.nc'
output_file = 'Data_noaa_copernicus/Copernicus/coper_miicescombine.nc'


combine_and_fill_nan(input_file1, input_file2, output_file)
