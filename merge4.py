"""
import pandas as pd

df = pd.read_csv('original_UW_data/daily_avgMI_eez.csv')
df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df[['Date', 'Lon', 'Lat', 'Temperature']]

# Save the DataFrame to a new CSV file
output_file = 'original_UW_data/daily_avgMI_eez.csv'
df.to_csv(output_file, index=False)
"""
import pandas as pd
df_ices=pd.read_csv('All_work_data/ICES/icesmerged_eez.csv')
df_mi=pd.read_csv('All_work_data/ICESMI/MI_data/original_UW_data/daily_avgMI_eez.csv')

icesmi_eez_data = pd.concat([df_ices, df_mi], ignore_index=True)
icesmi_eez_data.to_csv('All_work_data/ICESMI/icesmi_eez_data.csv', index=None)
#df=pd.read_csv('merged_data\\icesmi_merged_data.csv')

df_icesmi = pd.read_csv('All_work_data/ICESMI/icesmi_eez_data.csv')

# Round off Lon and Lat columns to two decimal points
df_icesmi['Lon'] = df_icesmi['Lon'].round(2)
df_icesmi['Lat'] = df_icesmi['Lat'].round(2)

"""
Final data file after merging icesmi
"""
# Save the modified DataFrame to a new CSV file
output_file = 'All_work_data/ICESMI/icesmi_eez_data.csv'
df_icesmi.to_csv(output_file, index=False)
#################################################################
"""
Spatial Plotting of the merged data
"""
import cartopy.crs as ccrs
import cartopy as cart
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


df=pd.read_csv('All_work_data/ICESMI/icesmi_eez_data.csv')

df['Date'] = pd.to_datetime(df['Date'])
fig = plt.figure(figsize=[18,10])
ax = plt.axes(projection=ccrs.PlateCarree())

plt.scatter(df['Lon'], df['Lat'], s=10, color='darkgoldenrod', marker='*',
             transform=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND,zorder=100, edgecolor='k')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

plt.ylabel ('Latitude')
plt.xlabel ('Longitude')
# Save the single plot as an image file
#plt.savefig(f"ices_mi_datapoints.png")

plt.show()
