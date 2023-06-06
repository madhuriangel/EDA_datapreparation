"""
Merging of ICES data
ICES_Ocean_pump_data, is not included as it doesn't have temperature values at depth=0

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import cartopy.crs as ccrs
import cartopy as cart
import cartopy.feature as cfeature

df1 = pd.read_csv('All_work_data/ICES/ICES_bottlelowresctd/bottle_lowres_ctd/bottle_lowres_temp.csv')
df2 = pd.read_csv('All_work_data/ICES/ICES_Expendabledatac/xbt_data/xbt_data_surfaceonly.csv')
df3 = pd.read_csv('All_work_data/ICES/ICES_high_resolution/high_res_ctd/high_res_data.csv')
df4 = pd.read_csv('All_work_data/ICES/ICES_OceanSurfacedat/Surface_data/oceansurfacedat_alter.csv')

#Concantenating all the ICES data
ices_merged_data = pd.concat([df1, df2, df3, df4], ignore_index=True)

#To save
#ices_merged_data.to_csv('All_work_data/ICES/ices_merged_data.csv',index=None)

#If starting from here, can use the file
#ices_merged_data=pd.read_csv('All_work_data/ICES/ices_merged_data.csv')

#Finding the blank or null rows
blank_rows=ices_merged_data[ices_merged_data['Temperature'].isnull()]

# Remove the blank rows from the DataFrame
#Updated dataframe with no null values
ices_merged_data = ices_merged_data.dropna(subset=['Temperature'])

#DUPLICATE VALUES
# Identify duplicates based on all columns (date, lon, lat, temperature)
duplicates = ices_merged_data[ices_merged_data.duplicated(keep=False)]#935 duplicates

# Print duplicate rows
if not duplicates.empty:
    print("Duplicate rows found:")
    print(duplicates)
else:
    print("No duplicate rows found.")
    
#Remove the duplicate values
# Remove duplicates based on all columns (date, lon, lat, temperature)
ices_merged_data_cleaned = ices_merged_data.drop_duplicates(keep='first')


# Save the DataFrame without duplicates to a new CSV file
#ices_merged_data_cleaned.to_csv('All_work_data/ICES/ices_merged_cleaned.csv',index=None)

df=pd.read_csv('All_work_data/ICES/ices_merged_cleaned.csv')

#Check for outliers in the temperature column using a boxplot
plt.figure(figsize=[18,10])
sns.set(style="darkgrid")
plt.rcParams.update({'font.size': 20, 'font.weight': 'bold'})
sns.boxplot(x=df['Temperature'],width=0.5,linewidth=2,color='darkgoldenrod')
plt.xlabel('Temperature', fontsize=14, fontweight='bold')
plt.title('Boxplot of Temperature', fontsize=14, fontweight='bold')
plt.show()


# Create a scatter plot of the "Temperature" column with a reference line
#With outliers
import os
# Calculate z-scores for the "Temperature" column
z_scores = (df['Temperature'] - df['Temperature'].mean()) / df['Temperature'].std()

# Define a threshold for outliers (e.g., z-score > 3)
outlier_threshold = 3

# Identify outliers based on the threshold
outliers = df[abs(z_scores) > outlier_threshold]

# Create the scatter plot with outliers in a different color
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['Temperature'], color='darkgoldenrod', label='Temperature')
plt.scatter(outliers.index, outliers['Temperature'], color='red', label='Outliers')
plt.axhline(y=df['Temperature'].mean(), color='blue', linestyle='--', label='Mean')
plt.xlabel('Index', fontsize=14, fontweight='bold')
plt.ylabel('Temperature', fontsize=14, fontweight='bold')
plt.title('Scatter Plot - Temperature', fontsize=14, fontweight='bold')
plt.legend()

# Save the plot in the specified folder
output_folder = 'All_work_data/ICES/plots'
filename = 'Outlier_Temperature.png'
filepath = os.path.join(output_folder, filename)
plt.savefig(filepath)
plt.close()

# Show the plot
plt.show()

# Remove the outliers from the DataFrame
df_cleaned = df.drop(outliers.index)

# Create a scatter plot of the "Temperature" column with a reference line
#After removing outliers
#plt.figure(figsize=(10, 6))
# plt.scatter(df_cleaned.index, df_cleaned['Temperature'], color='darkgoldenrod', label='Temperature')
# plt.axhline(y=df_cleaned['Temperature'].mean(), color='red', linestyle='--', label='Mean')
# plt.xlabel('Index',fontsize=14, fontweight='bold')
# plt.ylabel('Temperature',fontsize=14, fontweight='bold')
# plt.title('Scatter Plot - Temperature',fontsize=14, fontweight='bold')
# plt.legend()
# plt.show()
import geopandas as gpd
shapefile_path='eez/eez.shp'
eez=gpd.read_file(shapefile_path)

"""
   Filtering the df_cleaned using the shapefile Exclusive economic zone (EEZ)
   Input file:df_cleaned
   Output file:icesmerged_eez.csv
"""
# Convert the data to a geopandas GeoDataFrame
data_geo = gpd.GeoDataFrame(df_cleaned, geometry=gpd.points_from_xy(df_cleaned['Lon'], df_cleaned['Lat']))

# Assign CRS to the data_geo GeoDataFrame
data_geo.crs = 'EPSG:4326'

# Reproject the data_geo GeoDataFrame to match the CRS of the eez GeoDataFrame
data_geo = data_geo.to_crs(eez.crs)

# Perform spatial join to filter data within the shapefile boundaries
filtered_data = gpd.sjoin(data_geo, eez, predicate='within')

# Reset the index of the filtered_data GeoDataFrame
filtered_data = filtered_data.reset_index(drop=True)

# Drop unnecessary columns from the filtered data
filtered_data = filtered_data.drop(columns=['index_right'])

# Save only the original data columns to a new CSV file
filtered_data[df_cleaned.columns].to_csv('All_work_data/ICES/icesmerged_eez.csv', index=False)

df_eez=pd.read_csv('All_work_data/ICES/icesmerged_eez.csv')

fig = plt.figure(figsize=[18,10])
ax = plt.axes(projection=ccrs.PlateCarree())

plt.scatter(df_eez['Lon'], df_eez['Lat'], s=10, color='darkgoldenrod', marker='*',
             transform=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND,zorder=100, edgecolor='k')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

plt.ylabel ('Latitude')
plt.xlabel ('Longitude')

plt.show()



"""
Following code is the mm/yyyy wise plot
"""
# Convert the Date column to pandas datetime format
df_eez['Date'] = pd.to_datetime(df_eez['Date'])

# Extract year and month from the Date column
df_eez['Year'] = df_eez['Date'].dt.year
df_eez['Month'] = df_eez['Date'].dt.month

# Group the data by year and month
grouped_data = df_eez.groupby(['Year', 'Month'])

# Define the bounding box
bounding_box = [-14, -5, 49, 56]

# Define the figure size (width, height) in inches
figure_size = (15, 8)

# Iterate over each group to create subplots
for (year, month), group in grouped_data:
    # Create a subplot with a map using Cartopy and set the figure size
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Set the extent of the map based on the bounding box
    ax.set_extent(bounding_box)

    # Create a scatter plot for each group
    scatter = ax.scatter(
        group['Lon'],
        group['Lat'],
        c=group['Temperature'],
        cmap='coolwarm',
        transform=ccrs.PlateCarree()
    )

    # Annotate each data point
    #for index, row in group.iterrows():
        #ax.annotate(
            #text=row['Date'].strftime('%Y-%m-%d'),
           # xy=(row['Longitude [degrees_east]'], row['Latitude [degrees_north]']),
            #xytext=(5, 0),
            #textcoords='offset points',
            #transform=ccrs.PlateCarree()
        #)

   
    
    ax.set_title(f"ICES Temperature- {month}/{year}", fontsize=15, fontweight='bold')

    output_folder = 'All_work_data/ICES/plots'
    # Save the subplot as an image file
    filename = f"ICESTemperature{month:02d}_{year}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)

     # Show the current subplot
    plt.show()

