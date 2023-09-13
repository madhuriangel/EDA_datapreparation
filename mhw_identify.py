import numpy as np
import xarray as xr
"""
This code deals with identifying MHW according to the Hobday et al 2016 definition
Two data files combinefile.nc :NOAA, MI, ICES prepared data of resolution 0.25, 1982-2022 daily data, with sea surface temperature variable as 'sst'
       coper_miicescombine.nc :Copernicus, MI, ICES prepared data of resolution 0.05, 1982-2020 daily data with sea surface temperature variable as 'analysed_sst'
"""

#data = xr.open_dataset('Data_collection/noaa_avhrr/combinefile.nc')
data = xr.open_dataset('Data_collection/Copernicus/coper_miicescombine.nc')

# Define the criteria for a marine heatwave
def detect_marine_heatwaves(sst, duration_threshold=5, percentile_threshold=90, gap_threshold=2):
    # Initialize an array for labels (1 for MHW, 0 for non-MHW)
    labels = np.zeros(sst.shape, dtype=int)
    
    for lat_idx, lat in enumerate(data['lat']):
        for lon_idx, lon in enumerate(data['lon']):
            sst_grid = sst[:, lat_idx, lon_idx]  # Extract SST time series for a grid cell
            
            # Calculate the 90th percentile based on the 30-year baseline period (1982-2012), including leap years
            baseline_start = 0  # Index for the start of the baseline period
            baseline_end = baseline_start + (30 * 365) + 8  # 8 extra days for leap years
            baseline = sst_grid[baseline_start:baseline_end]
            percentile = np.percentile(baseline, percentile_threshold)
            
            # Initialize variables to track MHW events
            mhw_start = None
            mhw_duration = 0
            
            # Loop through time dimension
            for t in range(len(sst_grid)):
                if sst_grid[t] > percentile:
                    if mhw_start is None:
                        mhw_start = t
                    mhw_duration += 1
                else:
                    if mhw_duration >= duration_threshold:
                        labels[mhw_start:t, lat_idx, lon_idx] = 1  # Label as MHW
                    mhw_start = None
                    mhw_duration = 0
            
            # Consider gap threshold for continuous events
            if mhw_duration >= duration_threshold and t - mhw_start <= gap_threshold:
                labels[mhw_start:t + 1, lat_idx, lon_idx] = 1  # Label as MHW
    
    return labels

#sst_data = data['sst'].values #NOAA_MI_ICES data
sst_data = data['analysed_sst'].values # Extract SST values COPERNICUS_MI_ICES data
labels = detect_marine_heatwaves(sst_data)

labeled_data = data.copy()
labeled_data['mhw_labels'] = (('time', 'lat', 'lon'), labels)
#labeled_data.to_netcdf('noaami_labeled_data.nc')#This is for NOAA_MI_ICES data
labeled_data.to_netcdf('copernicusmi_labeled_data.nc')#This is for copernicus_MI_ICES data
