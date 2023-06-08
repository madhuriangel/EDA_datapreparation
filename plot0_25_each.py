import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

data = pd.read_csv('All_work_data/ICESMI/icesmi_eez_data.csv')

lon = data['Lon']
lat = data['Lat']
temp = data['Temperature']
date_column = data['Date']

data['Date'] = pd.to_datetime(data['Date'])

data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

unique_dates = data[['Month', 'Year']].drop_duplicates()

# Define the bounding box coordinates
lon_min, lon_max, lat_min, lat_max = -14, -5, 49, 56

resolution = 0.25

# Creating a meshgrid for longitude and latitude
lon_range = np.arange(lon_min, lon_max + resolution, resolution)
lat_range = np.arange(lat_min, lat_max + resolution, resolution)
lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

# Iterate over each month/year combination
for _, row in unique_dates.iterrows():
    month, year = row['Month'], row['Year']
    
    data_month = data[(data['Month'] == month) & (data['Year'] == year)]
    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.plot(lon_grid, lat_grid, 'k.', markersize=3, transform=ccrs.PlateCarree())
    scatter = ax.scatter(data_month['Lon'], data_month['Lat'], c=data_month['Temperature'], cmap='coolwarm', s=16, alpha=0.5,
                         transform=ccrs.PlateCarree())
   ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())


    cbar = plt.colorbar(scatter, label='Temperature (°C)', cmap='coolwarm', boundaries=np.linspace(temp.min(), temp.max(), 10))
    ax.set_title(f'{month}/{year}')

    # Save the plot to a file
    #plt.savefig(f'plot_{month}_{year}.png')
    #plt.close(fig)
    plt.show()
