#This is with two decimal points
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

data = pd.read_csv('All_work_data/ICESMI/icesmi_eez_data.csv')
lon = data['Lon']
lat = data['Lat']
temp = data['Temperature']

lon_min, lon_max, lat_min, lat_max = -14, -5, 49, 56
resolution = 0.25

lon_range = np.arange(lon_min, lon_max + resolution, resolution)
lat_range = np.arange(lat_min, lat_max + resolution, resolution)
lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

fig = plt.figure(figsize=(20, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
ax.plot(lon_grid, lat_grid, 'k.', markersize=5, transform=ccrs.PlateCarree())
scatter = ax.scatter(lon, lat, c=temp, cmap='hot', s=10, alpha=0.5, transform=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
cbar = plt.colorbar(scatter, label='Temperature (°C)', cmap='hot', boundaries=np.linspace(temp.min(), temp.max(), 10))

#plt.savefig('plot_0.25p.png')
plt.show()
