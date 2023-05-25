import pandas as pd
import numpy as np

# Define the grid resolution and bounding box
resolution = 0.25
lon_min, lon_max = -14, -5
lat_min, lat_max = 49, 56

# Load the CSV data
data = pd.read_csv('All_work_data/ICESMI/icesmidata_eez.csv')

# Filter data within the bounding box
data = data[(data['Lon'] >= lon_min) & (data['Lon'] <= lon_max) &
            (data['Lat'] >= lat_min) & (data['Lat'] <= lat_max)]

# Create a date range from 1982 to 2022
start_date = pd.to_datetime('1982-01-01')
end_date = pd.to_datetime('2022-12-31')
dates = pd.date_range(start_date, end_date, freq='D')

# Create an empty gridded dataset
grid_lon = np.arange(lon_min, lon_max + resolution, resolution)
grid_lat = np.arange(lat_min, lat_max + resolution, resolution)
grid_data = np.empty((len(dates), len(grid_lat), len(grid_lon)))
grid_data.fill(np.nan)

# Perform inverse distance weighting interpolation
for i, date in enumerate(dates):
    date_data = data[data['Date'] == str(date.date())]
    if not date_data.empty:
        for j, lat in enumerate(grid_lat):
            for k, lon in enumerate(grid_lon):
                distances = np.sqrt((date_data['Lat'] - lat) ** 2 + (date_data['Lon'] - lon) ** 2)
                weights = 1 / distances
                interpolated_value = np.sum(date_data['sst'] * weights) / np.sum(weights)
                grid_data[i, j, k] = interpolated_value
    else:
        grid_data[i] = -999.0

# Create a multi-index DataFrame with date, lat, lon, and sst columns
grid_df = pd.DataFrame(grid_data.reshape(len(dates), -1), index=dates, columns=pd.MultiIndex.from_product([grid_lat, grid_lon], names=['Lat', 'Lon']))
grid_df.columns.names = ['Lat', 'Lon']
grid_df = grid_df.stack(['Lat', 'Lon']).reset_index()
grid_df.columns = ['Date', 'Lat', 'Lon', 'sst']

# Save the gridded dataset to a new CSV file
grid_df.to_csv('gridded_data.csv', index=False)
