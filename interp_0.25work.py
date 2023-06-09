import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import xarray as xr

"""
Using input file:icesmi_eez_data.csv
Output file: gridded_temperature_0.25data.nc
"""
data = pd.read_csv('All_work_data/ICESMI/icesmi_eez_data.csv', parse_dates=['Date'])

lon_min, lon_max, lat_min, lat_max = -14, -5, 49, 56
resolution = 0.25

# Create a date range from 1/1/1982 to 31/12/2022
start_date = pd.to_datetime('1982-01-01')
end_date = pd.to_datetime('2022-12-31')
dates = pd.date_range(start_date, end_date, freq='D')

lon = np.arange(lon_min, lon_max + resolution, resolution)
lat = np.arange(lat_min, lat_max + resolution, resolution)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Initialize an empty array to store temperature data
temperature_data = np.full((len(dates), len(lat), len(lon)), np.nan)

for i, date in enumerate(dates):
    # Filter data for the current date
    date_data = data[data['Date'] == date]

    # Calculate distances between grid points and data points
    distances = cdist(
        date_data[['Lon', 'Lat']],
        np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    ).reshape(len(date_data), len(lat), len(lon))

    # Checking if any data points are within the threshold distance,taking distance threshold as 2*resolution
    has_data = np.any(distances <= 2 * resolution, axis=0)

    if np.any(has_data):
        weights = 1 / distances

        # Set weights to 0 for grid points without data nearby
        weights[:, ~has_data] = 0

        # Calculating weighted temperature values
        weighted_temps = np.sum(weights * date_data['Temperature'].values[:, None, None], axis=0)

        # Normalizing weights to avoid division by zero
        normalized_weights = np.sum(weights, axis=0)
        normalized_weights[normalized_weights == 0] = 1  # Avoid division by zero
        weighted_temps /= normalized_weights

        # Assigning calculated temperatures to the corresponding grid points
        temperature_data[i][has_data] = weighted_temps[has_data]

# Creating xarray dataset
ds = xr.Dataset(
    {'Temperature': (['Date', 'Lat', 'Lon'], temperature_data)},
    coords={'Date': dates, 'Lat': lat, 'Lon': lon}
)

# Saving dataset as NetCDF file
ds.to_netcdf('All_work_data/ICESMI/gridded_temperature_0.25data.nc')
