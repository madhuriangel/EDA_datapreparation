import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import xarray as xr

# Read the CSV file
data = pd.read_csv('All_work_data/ICESMI/icesmi_eez_data.csv')

# Convert the 'Date' column to datetime with day-first format
#data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Convert the 'Date' column to the desired format
#data['Date'] = data['Date'].dt.strftime('%d/%m/%Y')

# Print the unique values in the 'Date' column
#unique_dates = data['Date'].unique()
#print(unique_dates)

#data.to_csv('All_work_data/ICESMI/icesmi_eez_data.csv',index=False)


# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Create a date range from 1/1/1982 to 31/12/2022
date_range = pd.date_range(start='01/01/1982', end='31/12/2022')

# Create a grid of longitude and latitude points
lon = np.arange(-14, -4.75, 0.25)
lat = np.arange(49, 56.25, 0.25)
lon_mesh, lat_mesh = np.meshgrid(lon, lat)

# Create an empty array to hold the gridded data
gridded_data = np.empty((len(date_range), len(lon), len(lat)))
gridded_data.fill(np.nan)

# Iterate over each date in the date range
for i, date in enumerate(date_range):
    # Filter the data for the current date
    date_data = data[data['Date'] == date]

    # Check if there is any data for the current date
    if not date_data.empty:
        # Interpolate temperature values using inverse distance weighting
        points = date_data[['Lon', 'Lat']].values
        values = date_data['Temperature'].values
        gridded_temperature = griddata(points, values, (lon_mesh, lat_mesh), method='linear')

        # Transpose the gridded_temperature array to match the shape of gridded_data
        gridded_temperature = gridded_temperature.T

        # Assign the interpolated temperature values to the gridded data array
        gridded_data[i, :, :] = gridded_temperature


# Create a DataArray using xarray
coords = {
    'Date': date_range,
    'Lon': lon,
    'Lat': lat
}
dims = ['Date', 'Lon', 'Lat']
temperature_data = xr.DataArray(gridded_data, coords=coords, dims=dims)

# Create a Dataset using xarray and add the temperature data variable
dataset = xr.Dataset({'Temperature': temperature_data})

# Save the dataset as an nc file
dataset.to_netcdf('gridded_temperature_data.nc')
