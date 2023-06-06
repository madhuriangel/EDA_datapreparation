import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Convert the Date column to pandas datetime format
#df['Date'] = pd.to_datetime(df['Date'])
df=pd.read_csv('All_work_data/ICESMI/icesmi_eez_data.csv')

# Convert the Date column to pandas datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract year and month from the Date column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Determine the global minimum and maximum temperature values
global_min_temp = df['Temperature'].min()
global_max_temp = df['Temperature'].max()

# Group the data by year
year_grouped_data = df.groupby('Year')

# Define the figure size (width, height) in inches
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
        # Get the corresponding axis for the current month
        ax = axes[month - 1]

        # Check if there is data for the current month
        if month in month_grouped_data.groups:
            group = month_grouped_data.get_group(month)
            # Sort the group data by Date
            group = group.sort_values('Date')

            # Create a scatter plot for each group
            ax.scatter(group['Date'].dt.day, group['Temperature'], color='darkgoldenrod', marker='o')

        # Set title (bold) and labels for the subplot
        ax.set_title(f"{year}-{month:02d}",fontsize=15, fontweight='bold')
        ax.set_xlabel('Day of the Month', fontsize=15, fontweight='bold')
        ax.set_ylabel('Temperature [degC]', fontsize=15, fontweight='bold')

        # Generate a complete range of dates for the current year and month
        date_range = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-{pd.Timestamp(year, month, 1).days_in_month}")

        # Create an array of uniformly spaced tick values
        tick_values = np.linspace(1, pd.Timestamp(year, month, 1).days_in_month, num=8, dtype=int)

        # Set xticks and xtick labels with a 45-degree rotation and right horizontal alignment
        ax.set_xticks(tick_values)
        ax.set_xticklabels(date_range[tick_values - 1].strftime('%Y-%m-%d'), rotation=45, ha='right', fontsize=15)

        # Set the y-axis limits based on the global minimum and maximum temperature values
        ax.set_ylim(global_min_temp, global_max_temp)

    # Adjust the space between subplots
    plt.tight_layout()

    # Save the single plot as an image file
    #plt.savefig(f"Temperature_{year}.png")

    # Show the single plot
    plt.show()
