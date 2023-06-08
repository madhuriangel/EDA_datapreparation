
"""
   About the code:
       Using the file generated from code mi_filter_data1.py, as input in this code
       input file: UW_data_r_Ireland.csv
       Output file: daily_avgMI_eez.csv

"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as cart
import cartopy.feature as cfeature
import seaborn as sns
import numpy as np

df=pd.read_csv("original_UW_data/UW_data_r_Ireland.csv")

# Identify duplicates based on all columns
duplicates = df[df.duplicated(keep=False)]
if not duplicates.empty:
    print("Duplicate rows found:")
    print(duplicates)
else:
    print("No duplicate rows found.")

#Simple spatial plot to check the data points
plt.figure(figsize=[18,10])
ax=plt.axes(projection=ccrs.PlateCarree())

plt.scatter(df.Lon, df.Lat, s=10,color='darkgoldenrod',marker='*',transform=ccrs.PlateCarree())

ax.coastlines(resolution='50m')
ax.add_feature(cart.feature.LAND, zorder=100,edgecolor='k')
plt.ylabel('Latitude')
plt.xlabel('Longitude')

"""
Next part calculation of daily averages over the temperature, longitude, and latitude
Creation of new files holding the avergaes for each day
Data is from 1994-2022#
"""
print("Start Date:",df.Datetime.min())#'2022-12-03 19:42:50.000'
print("End Date:",df.Datetime.max())#'1994-03-25 08:57:30.000'

df['Datetime']=pd.to_datetime(df['Datetime'])

#Group the data by year,month and day
grouped=df.groupby([df['Datetime'].dt.year,df['Datetime'].dt.month,df['Datetime'].dt.day])

#Calculate the daily avergaes of temperature,longitude,latitude
daily_averages=grouped.agg({'Temperature':'mean','Lon':'mean','Lat':'mean'})
daily_averages.index.names=['year','month','day']
daily_averages=daily_averages.reset_index()

#Saving the averaged values to new file
daily_averages=daily_averages.to_csv('original_UW_data/daily_averages.csv',index=None)

"""
Available dataset is from 1994/03 to 2022/12 (YYYY/MM)
Must have total 10533 days data, but we have 6464 days
1994/03 it has 306 days excluding January and February
There are 7 leap years(1996,2000,2004,2008,2012,2016,2020)
Total Calculation 1994/03 to 2022/12
28 * 365 + 7(leap years) + 306 (from the year 1994) = 10220 + 7 +306 =10533 days

Total missing data (10533-6464)=4069 (38.63%) data is missing as per days
"""

avg_data=pd.read_csv('original_UW_data/daily_averages.csv') #6464,6
shapefile_path='eez/eez.shp'
eez=gpd.read_file(shapefile_path)

"""
   Filtering the daily averages using the shapefile Exclusive economic zone (EEZ)
   Input file:daily_averages.csv
   Output file:daily_avgMI_eez.csv
"""
data_geo = gpd.GeoDataFrame(avg_data, geometry=gpd.points_from_xy(avg_data['Lon'], avg_data['Lat']))
data_geo.crs = 'EPSG:4326'
data_geo = data_geo.to_crs(eez.crs)
filtered_data = gpd.sjoin(data_geo, eez, predicate='within')
filtered_data = filtered_data.reset_index(drop=True)
filtered_data = filtered_data.drop(columns=['index_right'])

# Save only the original data columns to a new CSV file
filtered_data[avg_data.columns].to_csv('original_UW_data/daily_avgMI_eez.csv', index=False)

"""
Input file:daily_avgMI_eez.csv
Output: Plot for month/year having the daily data points
"""

avg_data_eez=pd.read_csv('All_work_data/ICESMI/MI_data/original_UW_data/daily_avgMI_eez.csv') #4783,7
avg_data_eez['date'] = pd.to_datetime(avg_data_eez[['year', 'month', 'day']])   

min_temp = avg_data_eez['Temperature'].min()
max_temp = avg_data_eez['Temperature'].max()

figure_height = 10

for year, month in avg_data_eez[['year', 'month']].drop_duplicates().values:
    monthly_data = avg_data_eez.loc[(avg_data_eez['year'] == year) & (avg_data_eez['month'] == month)]
    fig = plt.figure(figsize=(15, figure_height))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, zorder=1)
    ax.set_extent([-14, -5, 49, 56], crs=ccrs.PlateCarree())

    # Scatter plot the temperature data
    sc = ax.scatter(monthly_data['Lon'], monthly_data['Lat'], c=monthly_data['Temperature'], cmap='plasma',
                    vmin=min_temp, vmax=max_temp, transform=ccrs.PlateCarree(), s=120)
    im = ax.imshow(np.array([[np.nan]]), cmap='plasma', vmin=min_temp, vmax=max_temp)
    ax.set_title(f'Temperature- {month}/{year}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, label='Temperature (°C', shrink=0.8)  # Adjust the shrink parameter as needed
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.label.set_weight('bold')

    import os

    output_folder = 'All_work_data/ICESMI/MI_data/plots_mi'
    # Save the plot in the specified folder
    filename = f'Temperature_{year}_{month}.png'
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    plt.close()






