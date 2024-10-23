# Import required modules
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, date  # Import date for ordinal conversion
import time

# Marine heatwave modules
import alternew_hobday as mhw

# Set paths and filenames
nc_file='Data_noaa_copernicus/noaa_avhrr/noaa_icesmi_combinefile_FINAL.nc'
#nc_file = 'noaa_newfile/noaasst1982_2023_data.nc'
outpath = 'fin_trial/alter_hobday'  # Output directory

# Define climatological baseline period
clim_b = [1991, 2020]  # Updated climatology period

print('PROCESSING:')

# Load data
print('Loading data... ')
start = time.time()
ds = xr.open_dataset(nc_file)
end = time.time()
print(f"Data loaded in {end - start} seconds")

# Extract variables
sst = ds.sst.values
time_arr = ds.time.values
lat = ds.lat.values
lon = ds.lon.values

sst.shape  # Check dimensions of array [time, lat, lon]

# Convert time array to ordinal time array
Ly_set = False  # Required for mhw.detect
time_o = pd.date_range('1982-01-01', periods=len(time_arr), freq='D').to_pydatetime()
time_o = np.array([date.toordinal() for date in time_o])

# Initialize variables for MHW statistics
time_yr = list(range(1982, 2024))
i_which = range(sst.shape[2])  # Longitude index
j_which = range(sst.shape[1])  # Latitude index
mhw_total_events = np.NaN * np.zeros((len(j_which), len(i_which)))
mhw_count = np.NaN * np.zeros((len(time_yr), len(j_which), len(i_which)))
mhw_intensity = np.NaN * np.zeros((len(time_yr), len(j_which), len(i_which)))
mhw_duration = np.NaN * np.zeros((len(time_yr), len(j_which), len(i_which)))
mhw_count_tr = np.NaN * np.zeros((len(j_which), len(i_which)))
mhw_intensity_tr = np.NaN * np.zeros((len(j_which), len(i_which)))
mhw_duration_tr = np.NaN * np.zeros((len(j_which), len(i_which)))
ev_max_max = np.NaN * np.zeros((len(j_which), len(i_which)))
ev_max_dur = np.NaN * np.zeros((len(j_which), len(i_which)))
ev_max_start = np.empty((len(j_which), len(i_which)), dtype="<U10")
ev_max_end = np.empty((len(j_which), len(i_which)), dtype="<U10")#new addition
ev_dur_max = np.NaN * np.zeros((len(j_which), len(i_which)))
ev_dur_dur = np.NaN * np.zeros((len(j_which), len(i_which)))
ev_dur_start = np.empty((len(j_which), len(i_which)), dtype="<U10")
ev_dur_end = np.empty((len(j_which), len(i_which)), dtype="<U10")#new addition

# Dictionary to store the start and end dates of multiple events
#mhw_event_dates = {}

# Loop over every grid point, and compute MHW statistics
for i in i_which:
    start = time.time()
    print(f"Processing {i} of {len(i_which) - 1}")
    for j in j_which:
        # Process single SST time series
        sst1 = sst[:, j, i]
        # Skip cells with land, ice
        if np.logical_not(np.isfinite(sst1.sum())) or (sst1 < -1).sum() > 0:
            mhw_total_events[j, i] = 0
        else:
            # Detect MHWs
            mhws, clim = mhw.detect(time_o, sst1, climatologyPeriod=clim_b, pctile=90, Ly=Ly_set)
            # Perform annual averaging of statistics
            mhwBlock = mhw.blockAverage(time_o, mhws, clim, temp=sst1)
            # Store total MHW counts
            mhw_total_events[j, i] = mhwBlock['count'].sum()
            # Store additional MHW variables
            mhw_count[:, j, i] = mhwBlock['count'] #each yr how many mhw took place
            mhw_intensity[:, j, i] = mhwBlock['intensity_max']  # Annual mean of max MHW intensity
            mhw_duration[:, j, i] = mhwBlock['duration']
            # Computes means and trends
            mean, trend, dtrend = mhw.meanTrend(mhwBlock)
            # Store trend data
            mhw_count_tr[j, i] = trend['count']
            mhw_intensity_tr[j, i] = trend['intensity_max']
            mhw_duration_tr[j, i] = trend['duration']
            # Store start dates of strongest/longest events
            ev_m = np.argmax(mhws['intensity_max'])  # Find strongest (intensity_max) event
            ev_d = np.argmax(mhws['duration'])  # Find longest (duration) event
            # Store statistics for strongest (intensity_max) event
            ev_max_max[j, i] = mhws['intensity_max'][ev_m]# Maximum intensity of the strongest event
            ev_max_dur[j, i] = mhws['duration'][ev_m]# Duration of the strongest event
            ev_max_start[j, i] = mhws['date_start'][ev_m].strftime("%Y-%m-%d")# Start date of the strongest event
            ev_max_end[j, i] = mhws['date_end'][ev_m].strftime("%Y-%m-%d") # End date of the strongest event
            # Store statistics for longest (duration) event
            ev_dur_max[j, i] = mhws['intensity_max'][ev_d]# Maximum intensity of the longest event
            ev_dur_dur[j, i] = mhws['duration'][ev_d]# Duration of the longest event
            ev_dur_start[j, i] = mhws['date_start'][ev_d].strftime("%Y-%m-%d")# Start date of the longest event
            ev_dur_end[j, i] = mhws['date_end'][ev_d].strftime("%Y-%m-%d") # End date of the longest event
            
            # Initialize event storage for this grid point
            #mhw_event_dates[(j, i)] = {}
            
            # Loop over each year to extract and store start/end dates of all events
            #for year_index in range(len(time_yr)):
                #events = mhw.get_events_for_year(mhws, time_yr[year_index])
                #mhw_event_dates[(j, i)][time_yr[year_index]] = events
            
            
    end = time.time()
    print(f"Processed in {end - start} seconds")


# Convert start dates to ordinal integers before saving
ev_max_start_ordinal = np.array([[date.fromisoformat(d).toordinal() if d else np.nan for d in row] for row in ev_max_start])
ev_dur_start_ordinal = np.array([[date.fromisoformat(d).toordinal() if d else np.nan for d in row] for row in ev_dur_start])

# Convert end dates to ordinal integers before saving
ev_max_end_ordinal = np.array([[date.fromisoformat(d).toordinal() if d else np.nan for d in row] for row in ev_max_end])
ev_dur_end_ordinal = np.array([[date.fromisoformat(d).toordinal() if d else np.nan for d in row] for row in ev_dur_end])

# Create xarray dataset for processed results
ds_out = ds.drop_vars({'sst'})  # Follow format of input dataset, without sst
dim = ds.sst.dims  # Read dimension names
ds_out.attrs = {}  # Clear attributes
ds_out['time'] = time_yr  # Set new time coordinate

# Store new variables in dataset
ds_out['mhw_total_events'] = ((dim[-2], dim[-1]), mhw_total_events)
ds_out['mhw_count'] = (('time', dim[-2], dim[-1]), mhw_count)
ds_out['mhw_intensity'] = (('time', dim[-2], dim[-1]), mhw_intensity)
ds_out['mhw_duration'] = (('time', dim[-2], dim[-1]), mhw_duration)
ds_out['mhw_count_tr'] = ((dim[-2], dim[-1]), mhw_count_tr)
ds_out['mhw_intensity_tr'] = ((dim[-2], dim[-1]), mhw_intensity_tr)
ds_out['mhw_duration_tr'] = ((dim[-2], dim[-1]), mhw_duration_tr)

# Store statistics for strongest and longest events
ds_out['ev_max_max'] = ((dim[-2], dim[-1]), ev_max_max)
ds_out['ev_max_dur'] = ((dim[-2], dim[-1]), ev_max_dur)
ds_out['ev_max_start'] = ((dim[-2], dim[-1]), ev_max_start_ordinal)
ds_out['ev_max_end'] = ((dim[-2], dim[-1]), ev_max_end_ordinal)
ds_out['ev_dur_max'] = ((dim[-2], dim[-1]), ev_dur_max)
ds_out['ev_dur_dur'] = ((dim[-2], dim[-1]), ev_dur_dur)
ds_out['ev_dur_start'] = ((dim[-2], dim[-1]), ev_dur_start_ordinal)
ds_out['ev_dur_end'] = ((dim[-2], dim[-1]), ev_dur_end_ordinal)

# Set attributes of variables
ds_out['time'] = ds_out.time.assign_attrs(units="years", standard_name="time", long_name="calendar year", axis="T", calendar="proleptic_gregorian")
ds_out['mhw_total_events'] = ds_out.mhw_total_events.assign_attrs(units="1", standard_name="n/a", long_name="Total number of marine heatwaves detected", coverage_content_type="auxiliaryInformation")
ds_out['mhw_count'] = ds_out.mhw_count.assign_attrs(units="1", standard_name="n/a", long_name="Count of marine heatwave events in each year", coverage_content_type="auxiliaryInformation")
ds_out['mhw_intensity'] = ds_out.mhw_intensity.assign_attrs(units=ds.sst.attrs['units'], standard_name="n/a", long_name="Annual mean of maximum marine heatwave intensities in each year (as an anomaly w.r.t. seasonal climatology)", coverage_content_type="auxiliaryInformation")
ds_out['mhw_duration'] = ds_out.mhw_duration.assign_attrs(units="1", standard_name="n/a", long_name="Mean duration (in days) of marine heatwave events in each year", coverage_content_type="auxiliaryInformation")
ds_out['mhw_count_tr'] = ds_out.mhw_count_tr.assign_attrs(units="1", standard_name="n/a", long_name="Trend in annual counts of marine heatwave events (events/year)", coverage_content_type="auxiliaryInformation")
ds_out['mhw_intensity_tr'] = ds_out.mhw_intensity_tr.assign_attrs(units=ds.sst.attrs['units'], standard_name="n/a", long_name="Trend in annual mean of maximum of marine heatwave intensities (degree_C/year)", coverage_content_type="auxiliaryInformation")
ds_out['mhw_duration_tr'] = ds_out.mhw_duration_tr.assign_attrs(units="1", standard_name="n/a", long_name="Trend in annual mean duration of marine heatwaves (days/year)", coverage_content_type="auxiliaryInformation")

# Set attributes for strongest and longest event variables
ds_out['ev_max_max'] = ds_out.ev_max_max.assign_attrs(units=ds.sst.attrs['units'], standard_name="n/a", long_name="Maximum intensity (as an anomaly w.r.t. seasonal climatology) of grid-point largest maximum intensity marine heatwave", coverage_content_type="auxiliaryInformation")
ds_out['ev_max_dur'] = ds_out.ev_max_dur.assign_attrs(units="1", standard_name="n/a", long_name="Duration (in days) of grid-point largest maximum intensity marine heatwave", coverage_content_type="auxiliaryInformation")
ds_out['ev_max_start'] = ds_out.ev_max_start.assign_attrs(units="days since 0001-01-01", standard_name="n/a", long_name="Start date of grid-point largest maximum intensity marine heatwave (in ordinal days)", coverage_content_type="auxiliaryInformation")
ds_out['ev_max_end'] = ds_out.ev_max_end.assign_attrs(
    units="days since 0001-01-01",
    standard_name="n/a",
    long_name="End date of grid-point largest maximum intensity marine heatwave (in ordinal days)",
    coverage_content_type="auxiliaryInformation"
)
ds_out['ev_dur_max'] = ds_out.ev_dur_max.assign_attrs(units=ds.sst.attrs['units'], standard_name="n/a", long_name="Maximum intensity (as an anomaly w.r.t. seasonal climatology) of grid-point longest duration marine heatwave", coverage_content_type="auxiliaryInformation")
ds_out['ev_dur_dur'] = ds_out.ev_dur_dur.assign_attrs(units="1", standard_name="n/a", long_name="Duration (in days) of grid-point longest duration marine heatwave", coverage_content_type="auxiliaryInformation")
ds_out['ev_dur_start'] = ds_out.ev_dur_start.assign_attrs(units="days since 0001-01-01", standard_name="n/a", long_name="Start date of grid-point longest duration marine heatwave (in ordinal days)", coverage_content_type="auxiliaryInformation")
ds_out['ev_dur_end'] = ds_out.ev_dur_end.assign_attrs(
    units="days since 0001-01-01",
    standard_name="n/a",
    long_name="End date of grid-point longest duration marine heatwave (in ordinal days)",
    coverage_content_type="auxiliaryInformation"
)

# Set global attributes
ds_out.attrs['title'] = "Marine heatwave statistics for the Ireland region"
ds_out.attrs['summary'] = "Data generated for analysis of marine heatwave characteristics and trends"
ds_out.attrs['reference climatology'] = f"{clim_b[0]}-{clim_b[1]}"
ds_out.attrs['Conventions'] = "ACDD-1.3"

# Create output filename
outfile = outpath + '/' + 'mhw_stats_alternewhobday'

# Save dataset to NetCDF file (uncompressed)
print('Saving data (uncompressed)... ')
start = time.time()
ds_out.to_netcdf(outfile + '_uncompressed' + '.nc')
end = time.time()
print(f"Data saved in {end - start} seconds")

# Save compressed dataset to NetCDF file
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in ds_out.data_vars}
print('Saving data (compressed)... ')
start = time.time()
ds_out.to_netcdf(outfile + '.nc', encoding=encoding)
end = time.time()
print(f"Data saved in {end - start} seconds")
