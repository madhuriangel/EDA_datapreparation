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
date_column = data['Date']

# Convert date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Extract month and year from the date
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Get unique month/year combinations
unique_dates = data[['Month', 'Year']].drop_duplicates()

# Define the bounding box coordinates
lon_min, lon_max, lat_min, lat_max = -14, -5, 49, 56

# Define the grid resolution
resolution = 0.25

# Create a meshgrid for longitude and latitude
lon_range = np.arange(lon_min, lon_max + resolution, resolution)
lat_range = np.arange(lat_min, lat_max + resolution, resolution)
lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

# Iterate over each month/year combination
for _, row in unique_dates.iterrows():
    month, year = row['Month'], row['Year']
    
    # Filter the data for the specific month/year
    data_month = data[(data['Month'] == month) & (data['Year'] == year)]

    # Create a spatial plot with the meshgrid and data points
    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastline and country borders
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

    # Plot the meshgrid
    ax.plot(lon_grid, lat_grid, 'k.', markersize=3, transform=ccrs.PlateCarree())

    # Overlay the data points on top of the meshgrid
    scatter = ax.scatter(data_month['Lon'], data_month['Lat'], c=data_month['Temperature'], cmap='coolwarm', s=16, alpha=0.5,
                         transform=ccrs.PlateCarree())

    # Set the plot extent to the bounding box
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Add a colorbar with darker shades
    cbar = plt.colorbar(scatter, label='Temperature (°C)', cmap='coolwarm', boundaries=np.linspace(temp.min(), temp.max(), 10))

    # Set the plot title to the specific month/year
    ax.set_title(f'{month}/{year}')

    # Save the plot to a file
    #plt.savefig(f'plot_{month}_{year}.png')

    # Close the figure to clear the plot for the next iteration
    #plt.close(fig)
    plt.show()
