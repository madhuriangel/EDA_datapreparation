import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df['Date'] = pd.to_datetime(df['Date'])
df=pd.read_csv('All_work_data/ICESMI/icesmi_eez_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
global_min_temp = df['Temperature'].min()
global_max_temp = df['Temperature'].max()

year_grouped_data = df.groupby('Year')

figure_size = (20, 12)

# Iterate over each year to create a single plot with monthly subplots
for year, year_group in year_grouped_data:
    # Create a grid of subplots (3 rows x 4 columns) for each year
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=figure_size, sharex='col', sharey='row')
    axes = axes.flatten()
    # Group the year data by month
    month_grouped_data = year_group.groupby('Month')
    # Iterate over each month
    for month in range(1, 13):
        ax = axes[month - 1]
        # Check if there is data for the current month
        if month in month_grouped_data.groups:
            group = month_grouped_data.get_group(month)
            group = group.sort_values('Date')
            ax.scatter(group['Date'].dt.day, group['Temperature'], color='darkgoldenrod', marker='o')
 
        ax.set_title(f"{year}-{month:02d}",fontsize=15, fontweight='bold')
        ax.set_xlabel('Day of the Month', fontsize=15, fontweight='bold')
        ax.set_ylabel('Temperature [degC]', fontsize=15, fontweight='bold')
        date_range = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-{pd.Timestamp(year, month, 1).days_in_month}")
        tick_values = np.linspace(1, pd.Timestamp(year, month, 1).days_in_month, num=8, dtype=int)
        ax.set_xticks(tick_values)
        ax.set_xticklabels(date_range[tick_values - 1].strftime('%Y-%m-%d'), rotation=45, ha='right', fontsize=15)
        ax.set_ylim(global_min_temp, global_max_temp)

    plt.tight_layout()

    # Save the single plot as an image file
    #plt.savefig(f"Temperature_{year}.png")
    plt.show()
