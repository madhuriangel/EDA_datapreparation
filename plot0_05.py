#This is with two decimal points
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Read in the data file
data = pd.read_csv('All_work_data/ICESMI/icesmi_eez_data.csv')

# Extract the relevant columns
lon = data['Lon']
lat = data['Lat']
temp = data['Temperature']

# Define the bounding box coordinates
lon_min, lon_max, lat_min, lat_max = -14, -5, 49, 56

# Define the grid resolution
resolution = 0.05

# Create a meshgrid for longitude and latitude
lon_range = np.arange(lon_min, lon_max + resolution, resolution)
lat_range = np.arange(lat_min, lat_max + resolution, resolution)
lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

# Create a spatial plot with the meshgrid and data points
fig = plt.figure(figsize=(20, 12))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add coastline and country borders
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

# Plot the meshgrid
ax.plot(lon_grid, lat_grid, 'k.', markersize=3, transform=ccrs.PlateCarree())

# Overlay the data points on top of the meshgrid
scatter = ax.scatter(lon, lat, c=temp, cmap='hot', s=10, alpha=0.5, transform=ccrs.PlateCarree())

# Set the plot extent to the bounding box
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add a colorbar with darker shades
cbar = plt.colorbar(scatter, label='Temperature (°C)', cmap='hot', boundaries=np.linspace(temp.min(), temp.max(), 10))

#plt.savefig('plot_0.05p.png')

# Show the plot
plt.show()
