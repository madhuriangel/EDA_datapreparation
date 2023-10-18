import xarray as xr
import numpy as np

def combine_and_fill_nan(input_file1, input_file2, output_file):
    # Load the input netCDF files
    data1 = xr.open_dataset(input_file1)
    data2 = xr.open_dataset(input_file2)

    # Replace NaN values in data1 with corresponding values from data2
    data1['sst'] = xr.DataArray(
        np.where(np.isnan(data1['sst']), data2['sst'], data1['sst']),
        dims=data1['sst'].dims,
        coords=data1['sst'].coords,
        name='sst'
    )

    # Add metadata attributes to the variables
    data1['sst'].attrs['long_name'] = 'Sea Surface Temperature'
    data1['sst'].attrs['units'] = 'Celsius'

    # Add global metadata attributes to the dataset
    data1.attrs['title'] = 'Combined Sea Surface Temperature Data from ICES, MI and NOAA_AVHRR'
    data1.attrs['Source1'] = 'Marine Institute (MI)'
    data1.attrs['Source2'] = 'International Council for the Exploration of the Seas (ICES)'
    data1.attrs['Source3'] = 'National Oceanic and Atmospheric Administration (NOAA)'
    data1.attrs['University'] = 'Atlantic Technological University'
    data1.attrs['Department'] = 'Marine and Freshwater Research Centre'
    data1.attrs['history'] = 'Created using custom Python script'
    data1.attrs['Conventions'] = 'CF-1.8'
    data1.attrs['Temporal Resolution']  = 'Daily from 1/1/1982 to 31/12/2022'
    data1.attrs['Spatial Resolution'] = 'Gridded 0.25*0.25'

    # Save the modified data as a new netCDF file with metadata
    data1.to_netcdf(output_file)

# Call the function with your input and output file paths
input_file1 = 'Data_noaa_copernicus/noaa_avhrr/icesmigriddedsst_0.25data_standard.nc'
input_file2 = 'Data_noaa_copernicus/noaa_avhrr/final_noaasst_data.nc'
output_file = 'Data_noaa_copernicus/noaa_avhrr/noaa_icesmi_combinefile.nc'

combine_and_fill_nan(input_file1, input_file2, output_file)
