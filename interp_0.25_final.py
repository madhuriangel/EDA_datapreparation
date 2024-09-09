"""
This code is to perform IDW on MI+ICES DATA, to estimate values as per
the availability of data points around the grid else just keeping the gridpoints as NaN
"""
import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

data_path = 'All_work_data/ICESMI/icesmi_eez_data.csv'
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])

# Generate date range
start_date = '1982-01-01'
end_date = '2023-12-31'
dates = pd.date_range(start_date, end_date)

# Grid boundaries and resolution
lon_bounds = [-12.875, -4.125]
lat_bounds = [49.125, 55.875]
resolution = 0.25

# Generate grid points
lon_grid = np.arange(lon_bounds[0], lon_bounds[1] + resolution, resolution)
lat_grid = np.arange(lat_bounds[0], lat_bounds[1] + resolution, resolution)
lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T

# Define IDW function
def inverse_distance_weighting(x, y, z, xi, yi, power=2, epsilon=1e-12):
    # Calculate the distance with a small epsilon to avoid division by zero
    dist = np.sqrt((xi - x[:, None])**2 + (yi - y[:, None])**2) + epsilon
    # Handle zero distances by setting a very high weight for those points
    weights = np.where(dist < epsilon, 1.0, 1 / dist**power)
    weights /= weights.sum(axis=0)
    return np.dot(z, weights)

# Initialize array for the results
temperature_data = np.full((len(dates), len(lat_grid), len(lon_grid[0])), np.nan)

# Perform interpolation for each date
for i, date in enumerate(dates):
    day_data = df[df['Date'] == date]
    if not day_data.empty:
        tree = cKDTree(day_data[['Lon', 'Lat']])
        dist, idx = tree.query(grid_points, k=1, distance_upper_bound=2 * resolution)
        
        # Consider only valid points (i.e., within the range of the data)
        valid_points = np.isfinite(dist) & (dist < 2 * resolution)
        
        if valid_points.any():
            valid_grid_points = grid_points[valid_points]
            interpolated_values = inverse_distance_weighting(
                day_data['Lon'].values,
                day_data['Lat'].values,
                day_data['Temperature'].values,
                valid_grid_points[:, 0],
                valid_grid_points[:, 1]
            )
            
            # Assign interpolated values to the correct positions in the grid
            grid_indices = np.where(valid_points.reshape(lat_grid.shape))
            temperature_data[i, grid_indices[0], grid_indices[1]] = interpolated_values

# Create new xarray Dataset
ds = xr.Dataset(
    {
        'sst': (['time', 'lat', 'lon'], temperature_data)
    },
    coords={
        'time': dates,
        'lat': lat_grid[:, 0],
        'lon': lon_grid[0, :]
    }
)

# Save to new NetCDF
output_path = 'All_work_data/ICESMI/Final_icesmi_gridded0.25_t.nc'
ds.to_netcdf(output_path)
