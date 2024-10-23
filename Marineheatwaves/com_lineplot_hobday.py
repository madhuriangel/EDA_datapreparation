import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np

file1_path = 'fin_trial/mhw_stats_hobday.nc'
file2_path = 'fin_trial/alter_hobday/mhw_stats_alternewhobday.nc'
ds1 = xr.open_dataset(file1_path, decode_times=False)
ds2 = xr.open_dataset(file2_path, decode_times=False)

output_dir = 'fin_trial/combined_lineplots'
os.makedirs(output_dir, exist_ok=True)

def plot_combined_variable(variable1, variable2, ds1, ds2, var_name, units, label1, label2):
    """
    Plots the comparison of yearly mean values of a specified variable from two datasets (Hobday and Alter)
    as line plots, and saves the plot to the specified output directory.

    Parameters:
    ----------
    variable1 : np.ndarray
        Yearly mean values of the specified variable from the Hobday dataset.
    variable2 : np.ndarray
        Yearly mean values of the specified variable from the Alter dataset.
    ds1 : xarray.Dataset
        The Hobday dataset containing the variable and time coordinates.
    ds2 : xarray.Dataset
        The Alter dataset containing the variable and time coordinates.
    var_name : str
        Name of the variable being plotted (used for titles and labels).
    units : str
        Units of the variable (used for labeling y-axis).
    label1 : str
        Label for the first dataset (e.g., 'hobday').
    label2 : str
        Label for the second dataset (e.g., 'alter').

    Returns:
    -------
    Saves the comparison plot of the specified variable from both datasets to the output directory.
    """
    years1 = ds1['time'].values
    years2 = ds2['time'].values
    
    # Ensure both datasets have the same number of years
    if len(years1) != len(years2):
        raise ValueError("The two datasets must have the same number of years")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot first dataset
    ax.plot(years1, variable1, marker='o', label=f'{label1}', color='darkgoldenrod')
    
    # Plot second dataset
    ax.plot(years2, variable2, marker='o', label=f'{label2}', color='#042759')  # Changed color to 042759

    ax.set_title(f'Yearly {var_name}', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel(f'{var_name} ({units})', fontsize=14)

    ax.set_xticks(np.arange(1982, np.max(years1) + 1, 2))

    ax.set_yticks(np.arange(0, max(max(variable1), max(variable2)) + 1, 1))

    ax.legend()

    output_file = os.path.join(output_dir, f'{var_name}_combined.png')
    plt.savefig(output_file)
    plt.close()

mhw_intensity1 = ds1['mhw_intensity'].mean(dim=('lat', 'lon')).values
mhw_intensity2 = ds2['mhw_intensity'].mean(dim=('lat', 'lon')).values
mhw_duration1 = ds1['mhw_duration'].mean(dim=('lat', 'lon')).values
mhw_duration2 = ds2['mhw_duration'].mean(dim=('lat', 'lon')).values
mhw_count1 = ds1['mhw_count'].mean(dim=('lat', 'lon')).values
mhw_count2 = ds2['mhw_count'].mean(dim=('lat', 'lon')).values

# Plotting the combined variables for each variable
plot_combined_variable(mhw_intensity1, mhw_intensity2, ds1, ds2, 'MHW Intensity', '°C', 'hobday', 'alter')
plot_combined_variable(mhw_duration1, mhw_duration2, ds1, ds2, 'MHW Duration', 'Days', 'hobday', 'alter')
plot_combined_variable(mhw_count1, mhw_count2, ds1, ds2, 'MHW Count', 'Events', 'hobday', 'alter')
