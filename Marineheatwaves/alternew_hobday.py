
#ALTER HOBDAY CODE
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import stats
import scipy.ndimage as ndimage
from datetime import date


def detect(t, temp, climatologyPeriod=[None, None], pctile=99, windowHalfWidth=5, 
           smoothPercentile=True, smoothPercentileWidth=31, minDuration=5, 
           joinAcrossGaps=True, maxGap=4, maxPadLength=False, 
           alternateClimatology=False, Ly=False):
    """
    Detect marine heatwave events based on sea surface temperature (SST) time series.

    Parameters:
    ----------
    t : array-like
        Time vector in ordinal format (i.e., date.toordinal()).
    temp : array-like
        SST time series for a single grid point.
    climatologyPeriod : list, optional
        Start and end years defining the climatological baseline period, by default [None, None].
    pctile : float, optional
        Percentile threshold to define heatwave events, by default 99.
    windowHalfWidth : int, optional
        Half-width of the window for calculating climatology and threshold, by default 5 days.
    smoothPercentile : bool, optional
        If True, smooth the percentile threshold using a running average, by default True.
    smoothPercentileWidth : int, optional
        Width of the running average window for smoothing, by default 31 days.
    minDuration : int, optional
        Minimum duration (in days) for a heatwave event to be detected, by default 5 days.
    joinAcrossGaps : bool, optional
        If True, merges events separated by small gaps, by default True.
    joinAcrossGaps : bool, optional
        If True, the function will attempt to join consecutive MHW events that are 
        separated by gaps smaller than or equal to `maxGap`. The logic for merging 
        events is based on the SST values before and after the gap. Specifically:
        - For each gap, the 6-day mean SST prior to the gap and the 6-day mean SST 
          following the gap are computed.
        - If both the pre-gap and post-gap means exceed the climatological threshold 
          (e.g., 99th percentile) for marine heatwaves, the events are merged into 
          one continuous event.
        - If either mean is below the threshold, the events are kept separate.
        - The function repeats this check and merging process for all gaps smaller 
          than or equal to `maxGap` until no further merging is possible.
    maxGap : int, optional
        Maximum number of gap days allowed between events for merging, by default 4 days.
    maxPadLength : int or bool, optional
        If not False, limits the maximum number of consecutive missing values (NaNs) 
        that can be interpolated, by default False.
    alternateClimatology : bool or list, optional
        If a list, provides an alternate climatology as [tClim, tempClim], by default False.
    Ly : bool, optional
        If True, forces climatological smoothing even with NaN values, by default False.

    Returns:
    -------
    mhw : dict
        Contains information about detected marine heatwave events, including:
            - 'time_start', 'time_end', 'time_peak': Event start, end, and peak times (ordinal).
            - 'date_start', 'date_end', 'date_peak': Event start, end, and peak dates.
            - 'index_start', 'index_end', 'index_peak': Event start, end, and peak indices.
            - 'duration': Duration of each event (in days).
            - 'intensity_max', 'intensity_mean', 'intensity_var', 'intensity_cumulative': 
              Various measures of event intensity.
    clim : dict
        Climatology and thresholds, including:
            - 'thresh': Temperature thresholds for detecting events.
            - 'seas': Climatological seasonal cycle (smoothed mean).
            - 'missing': Boolean mask of missing data in the time series.
    """
    
    # Initialize MHW output variable
    mhw = {
        'time_start': [], 'time_end': [], 'time_peak': [],
        'date_start': [], 'date_end': [], 'date_peak': [],
        'index_start': [], 'index_end': [], 'index_peak': [],
        'duration': [], 'intensity_max': [], 'intensity_mean': [],
        'intensity_var': [], 'intensity_cumulative': [],
        'intensity_max_relThresh': [],
        'intensity_mean_relThresh': [],
        'intensity_var_relThresh':[],
        'intensity_cumulative_relThresh': [],
        'intensity_max_abs':[],
        'intensity_mean_abs':[],
        'intensity_var_abs': [],
        'intensity_cumulative_abs': []
    }

    # Time and dates vectors
    T = len(t)
    year = np.zeros((T))
    month = np.zeros((T))
    day = np.zeros((T))
    doy = np.zeros((T))
    for i in range(T):
        year[i] = date.fromordinal(t[i]).year
        month[i] = date.fromordinal(t[i]).month
        day[i] = date.fromordinal(t[i]).day

    # Generate leap year calendar
    year_leapYear = 2012  # Chosen reference leap year
    t_leapYear = np.arange(date(year_leapYear, 1, 1).toordinal(), 
                           date(year_leapYear, 12, 31).toordinal() + 1)
    dates_leapYear = [date.fromordinal(tt.astype(int)) for tt in t_leapYear]
    
    month_leapYear = np.zeros((len(t_leapYear)))
    day_leapYear = np.zeros((len(t_leapYear)))
    doy_leapYear = np.zeros((len(t_leapYear)))
    for tt in range(len(t_leapYear)):
        month_leapYear[tt] = date.fromordinal(t_leapYear[tt]).month
        day_leapYear[tt] = date.fromordinal(t_leapYear[tt]).day
        doy_leapYear[tt] = t_leapYear[tt] - date(date.fromordinal(t_leapYear[tt]).year, 1, 1).toordinal() + 1

    # Assign day of year (doy) for each time point
    for tt in range(T):
        doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])]

    feb28 = 59
    feb29 = 60

    # Climatology period setup
    if (climatologyPeriod[0] is None) or (climatologyPeriod[1] is None):
        climatologyPeriod[0] = year[0]
        climatologyPeriod[1] = year[-1]

    if alternateClimatology:
        tClim = alternateClimatology[0]
        tempClim = alternateClimatology[1]
        TClim = len(tClim)
        yearClim = np.zeros((TClim))
        monthClim = np.zeros((TClim))
        dayClim = np.zeros((TClim))
        doyClim = np.zeros((TClim))
        for i in range(TClim):
            yearClim[i] = date.fromordinal(tClim[i]).year
            monthClim[i] = date.fromordinal(tClim[i]).month
            dayClim[i] = date.fromordinal(tClim[i]).day
            doyClim[i] = doy_leapYear[(month_leapYear == monthClim[i]) * (day_leapYear == dayClim[i])]
    else:
        tempClim = temp.copy()
        TClim = np.array([T]).copy()[0]
        yearClim = year.copy()
        monthClim = month.copy()
        dayClim = day.copy()
        doyClim = doy.copy()

    # Option to limit maximum consecutive NaNs
    if maxPadLength:
        temp = pad(temp, maxPadLength=maxPadLength)
        tempClim = pad(tempClim, maxPadLength=maxPadLength)

    # Climatology and threshold
    lenClimYear = 366
    clim_start = np.where(yearClim == climatologyPeriod[0])[0][0]
    clim_end = np.where(yearClim == climatologyPeriod[1])[0][-1]
    thresh_climYear = np.nan * np.zeros(lenClimYear)
    seas_climYear = np.nan * np.zeros(lenClimYear)
    clim = {'thresh': np.nan * np.zeros(TClim), 'seas': np.nan * np.zeros(TClim)}

    for d in range(1, lenClimYear + 1):
        if d == feb29:
            continue
        tt0 = np.where(doyClim[clim_start:clim_end + 1] == d)[0]
        if len(tt0) == 0:
            continue
        tt = np.array([])

        for w in range(-windowHalfWidth, windowHalfWidth + 1):
            tt = np.append(tt, clim_start + tt0 + w)
        tt = tt[tt >= 0]
        tt = tt[tt < TClim]
        thresh_climYear[d - 1] = np.nanpercentile(tempClim[tt.astype(int)], pctile)
        seas_climYear[d - 1] = np.nanmean(tempClim[tt.astype(int)])

    # Handle Feb 29th with interpolation
    thresh_climYear[feb29 - 1] = 0.5 * thresh_climYear[feb29 - 2] + 0.5 * thresh_climYear[feb29]
    seas_climYear[feb29 - 1] = 0.5 * seas_climYear[feb29 - 2] + 0.5 * seas_climYear[feb29]

    if smoothPercentile:
        valid = ~np.isnan(thresh_climYear)
        thresh_climYear[valid] = runavg(thresh_climYear[valid], smoothPercentileWidth)
        valid = ~np.isnan(seas_climYear)
        seas_climYear[valid] = runavg(seas_climYear[valid], smoothPercentileWidth)

    clim['thresh'] = thresh_climYear[doy.astype(int) - 1]
    clim['seas'] = seas_climYear[doy.astype(int) - 1]

    clim['missing'] = np.isnan(temp)
    temp[np.isnan(temp)] = clim['seas'][np.isnan(temp)]

    exceed_bool = temp - clim['thresh']
    exceed_bool[exceed_bool <= 0] = False
    exceed_bool[exceed_bool > 0] = True
    exceed_bool[np.isnan(exceed_bool)] = False
    events, n_events = ndimage.label(exceed_bool)

    # Detecting events based on exceedance
    for ev in range(1, n_events + 1):
        event_duration = (events == ev).sum()
        if event_duration < minDuration:
            continue
        mhw['time_start'].append(t[np.where(events == ev)[0][0]])
        mhw['time_end'].append(t[np.where(events == ev)[0][-1]])
        mhw['index_start'].append(np.where(events == ev)[0][0])
        mhw['index_end'].append(np.where(events == ev)[0][-1])

    # Gap merging logic
    if joinAcrossGaps:
        gaps = np.array(mhw['time_start'][1:]) - np.array(mhw['time_end'][0:-1]) - 1
        
        if len(gaps) > 0:
            while gaps.min() <= maxGap:
                ev = np.where(gaps <= maxGap)[0][0]
                
                # Calculate the pre-gap and post-gap 6-day means
                pre_gap_start = mhw['index_start'][ev] - 5  # 6-day window before gap including gap day
                pre_gap_end = mhw['index_start'][ev]  # Include the gap day
                pre_gap_mean = np.mean(temp[pre_gap_start:pre_gap_end + 1])
                
                post_gap_start = mhw['index_end'][ev + 1]  # Include the gap day
                post_gap_end = mhw['index_end'][ev + 1] + 5  # 6-day window after gap including gap day
                post_gap_mean = np.mean(temp[post_gap_start:post_gap_end + 1])

                # Check if both pre and post 6-day means exceed the 99th percentile
                if pre_gap_mean > clim['thresh'][pre_gap_start] and post_gap_mean > clim['thresh'][post_gap_start]:
                    # If both exceed the threshold, merge the events
                    mhw['time_end'][ev] = mhw['time_end'][ev + 1]
                    del mhw['time_start'][ev + 1]
                    del mhw['time_end'][ev + 1]
                else:
                    # If either does not exceed, keep them as separate events
                    break
                
                # Recalculate gaps
                gaps = np.array(mhw['time_start'][1:]) - np.array(mhw['time_end'][0:-1]) - 1
                if len(gaps) == 0:
                    break
            
            
    mhw['n_events'] = len(mhw['time_start'])

    for ev in range(mhw['n_events']):
        mhw['date_start'].append(date.fromordinal(mhw['time_start'][ev]))
        mhw['date_end'].append(date.fromordinal(mhw['time_end'][ev]))
        tt_start = np.where(t == mhw['time_start'][ev])[0][0]
        tt_end = np.where(t == mhw['time_end'][ev])[0][0]
        temp_mhw = temp[tt_start:tt_end + 1]
        thresh_mhw = clim['thresh'][tt_start:tt_end + 1]
        seas_mhw = clim['seas'][tt_start:tt_end + 1]
        mhw_relSeas = temp_mhw - seas_mhw
        mhw_relSeas = temp_mhw - seas_mhw
        mhw_relThresh = temp_mhw - thresh_mhw
        mhw_abs = temp_mhw
        tt_peak = np.argmax(mhw_relSeas)
        mhw['time_peak'].append(mhw['time_start'][ev] + tt_peak)
        mhw['date_peak'].append(date.fromordinal(mhw['time_start'][ev] + tt_peak))
        mhw['index_peak'].append(tt_start + tt_peak)
        mhw['duration'].append(len(mhw_relSeas))
        mhw['intensity_max'].append(mhw_relSeas[tt_peak])
        mhw['intensity_mean'].append(mhw_relSeas.mean())
        mhw['intensity_var'].append(np.sqrt(mhw_relSeas.var()))
        mhw['intensity_cumulative'].append(mhw_relSeas.sum())
        mhw['intensity_max_relThresh'].append(mhw_relThresh[tt_peak])
        mhw['intensity_mean_relThresh'].append(mhw_relThresh.mean())
        mhw['intensity_var_relThresh'].append(np.sqrt(mhw_relThresh.var()))
        mhw['intensity_cumulative_relThresh'].append(mhw_relThresh.sum())
        mhw['intensity_max_abs'].append(mhw_abs[tt_peak])
        mhw['intensity_mean_abs'].append(mhw_abs.mean())
        mhw['intensity_var_abs'].append(np.sqrt(mhw_abs.var()))
        mhw['intensity_cumulative_abs'].append(mhw_abs.sum())

    return mhw, clim
"""
def get_events_for_year(mhws, year):
    events = []
    for ev in range(mhws['n_events']):
        event_year_start = mhws['date_start'][ev].year
        event_year_end = mhws['date_end'][ev].year
        
        # If the MHW event started in the specified year
        if event_year_start == year:
            start_date = mhws['date_start'][ev].strftime("%Y-%m-%d")
            end_date = mhws['date_end'][ev].strftime("%Y-%m-%d")
            events.append((start_date, end_date))
        
        # If the MHW event ended in the specified year (spanning multiple years)
        elif event_year_end == year:
            start_date = mhws['date_start'][ev].strftime("%Y-%m-%d")
            end_date = mhws['date_end'][ev].strftime("%Y-%m-%d")
            events.append((start_date, end_date))
            
    return events
"""
def blockAverage(t, mhw, clim=None, blockLength=1, removeMissing=False, temp=None):
    """
    Compute marine heatwave (MHW) properties averaged over blocks of time.

    Parameters:
    ----------
    t : array-like
        Time vector in ordinal format.
    mhw : dict
        Marine heatwaves detected using the detect function.
    clim : dict, optional
        Climatology data, used to check for missing values, by default None.
    blockLength : int, optional
        Length of time blocks in years, by default 1 year.
    removeMissing : bool, optional
        If True, removes blocks with missing temperature data, by default False.
    temp : array-like, optional
        Temperature time series, used for additional statistics, by default None.

    Returns:
    -------
    mhwBlock : dict
        Block-averaged marine heatwave properties:
            - 'count', 'duration', 'intensity_max', 'intensity_mean', etc.
    """
 #
    # Time and dates vectors, and calculate block timing
    #

    # Generate vectors for year, month, day-of-month, and day-of-year
    T = len(t)
    year = np.zeros((T))
    month = np.zeros((T))
    day = np.zeros((T))
    for i in range(T):
        year[i] = date.fromordinal(t[i]).year
        month[i] = date.fromordinal(t[i]).month
        day[i] = date.fromordinal(t[i]).day

    # Number of blocks, round up to include partial blocks at end
    years = np.unique(year)
    nBlocks = np.ceil((years.max() - years.min() + 1) / blockLength).astype(int)


    sw_temp = None
    if temp is not None:
        sw_temp = True
    else:
        sw_temp = False

    #
    # Initialize MHW output variable
    #

    mhwBlock = {}
    mhwBlock['count'] = np.zeros(nBlocks)
    mhwBlock['duration'] = np.zeros(nBlocks)
    mhwBlock['intensity_max'] = np.zeros(nBlocks)
    mhwBlock['intensity_max_max'] = np.zeros(nBlocks)
    mhwBlock['intensity_mean'] = np.zeros(nBlocks)
    mhwBlock['intensity_cumulative'] = np.zeros(nBlocks)
    mhwBlock['intensity_var'] = np.zeros(nBlocks)
    mhwBlock['intensity_max_relThresh'] = np.zeros(nBlocks)
    mhwBlock['intensity_mean_relThresh'] = np.zeros(nBlocks)
    mhwBlock['intensity_cumulative_relThresh'] = np.zeros(nBlocks)
    mhwBlock['intensity_var_relThresh'] = np.zeros(nBlocks)
    mhwBlock['intensity_max_abs'] = np.zeros(nBlocks)
    mhwBlock['intensity_mean_abs'] = np.zeros(nBlocks)
    mhwBlock['intensity_cumulative_abs'] = np.zeros(nBlocks)
    mhwBlock['intensity_var_abs'] = np.zeros(nBlocks)
    mhwBlock['total_days'] = np.zeros(nBlocks)
    mhwBlock['total_icum'] = np.zeros(nBlocks)
    if sw_temp:
        mhwBlock['temp_mean'] = np.zeros(nBlocks)
        mhwBlock['temp_max'] = np.zeros(nBlocks)
        mhwBlock['temp_min'] = np.zeros(nBlocks)

    # Start, end, and centre years for all blocks
    mhwBlock['years_start'] = years[range(0, len(years), blockLength)]
    mhwBlock['years_end'] = mhwBlock['years_start'] + blockLength - 1
    mhwBlock['years_centre'] = 0.5*(mhwBlock['years_start'] + mhwBlock['years_end'])

    #
    # Calculate block averages
    #

    for i in range(mhw['n_events']):
        # Block index for year of each MHW (MHW year defined by start year)
        iBlock = np.where((mhwBlock['years_start'] <= mhw['date_start'][i].year) * (mhwBlock['years_end'] >= mhw['date_start'][i].year))[0][0]
        # Add MHW properties to block count
        mhwBlock['count'][iBlock] += 1
        mhwBlock['duration'][iBlock] += mhw['duration'][i]
        mhwBlock['intensity_max'][iBlock] += mhw['intensity_max'][i]
        mhwBlock['intensity_max_max'][iBlock] = np.max([mhwBlock['intensity_max_max'][iBlock], mhw['intensity_max'][i]])
        mhwBlock['intensity_mean'][iBlock] += mhw['intensity_mean'][i]
        mhwBlock['intensity_cumulative'][iBlock] += mhw['intensity_cumulative'][i]
        mhwBlock['intensity_var'][iBlock] += mhw['intensity_var'][i]
        mhwBlock['intensity_max_relThresh'][iBlock] += mhw['intensity_max_relThresh'][i]
        mhwBlock['intensity_mean_relThresh'][iBlock] += mhw['intensity_mean_relThresh'][i]
        mhwBlock['intensity_cumulative_relThresh'][iBlock] += mhw['intensity_cumulative_relThresh'][i]
        mhwBlock['intensity_var_relThresh'][iBlock] += mhw['intensity_var_relThresh'][i]
        mhwBlock['intensity_max_abs'][iBlock] += mhw['intensity_max_abs'][i]
        mhwBlock['intensity_mean_abs'][iBlock] += mhw['intensity_mean_abs'][i]
        mhwBlock['intensity_cumulative_abs'][iBlock] += mhw['intensity_cumulative_abs'][i]
        mhwBlock['intensity_var_abs'][iBlock] += mhw['intensity_var_abs'][i]
        if mhw['date_start'][i].year == mhw['date_end'][i].year: # MHW in single year
            mhwBlock['total_days'][iBlock] += mhw['duration'][i]
        else: # MHW spans multiple years
            year_mhw = year[mhw['index_start'][i]:mhw['index_end'][i]+1]
            for yr_mhw in np.unique(year_mhw):
                iBlock = np.where((mhwBlock['years_start'] <= yr_mhw) * (mhwBlock['years_end'] >= yr_mhw))[0][0]
                mhwBlock['total_days'][iBlock] += np.sum(year_mhw == yr_mhw)
        # NOTE: icum for a MHW is assigned to its start year, even if it spans mult. years
        mhwBlock['total_icum'][iBlock] += mhw['intensity_cumulative'][i]

    # Calculate averages
    count = 1.*mhwBlock['count']
    count[count==0] = np.nan
    mhwBlock['duration'] = mhwBlock['duration'] / count
    mhwBlock['intensity_max'] = mhwBlock['intensity_max'] / count
    mhwBlock['intensity_mean'] = mhwBlock['intensity_mean'] / count
    mhwBlock['intensity_cumulative'] = mhwBlock['intensity_cumulative'] / count
    mhwBlock['intensity_var'] = mhwBlock['intensity_var'] / count
    mhwBlock['intensity_max_relThresh'] = mhwBlock['intensity_max_relThresh'] / count
    mhwBlock['intensity_mean_relThresh'] = mhwBlock['intensity_mean_relThresh'] / count
    mhwBlock['intensity_cumulative_relThresh'] = mhwBlock['intensity_cumulative_relThresh'] / count
    mhwBlock['intensity_var_relThresh'] = mhwBlock['intensity_var_relThresh'] / count
    mhwBlock['intensity_max_abs'] = mhwBlock['intensity_max_abs'] / count
    mhwBlock['intensity_mean_abs'] = mhwBlock['intensity_mean_abs'] / count
    mhwBlock['intensity_cumulative_abs'] = mhwBlock['intensity_cumulative_abs'] / count
    mhwBlock['intensity_var_abs'] = mhwBlock['intensity_var_abs'] / count
    mhwBlock['intensity_max_max'][np.isnan(mhwBlock['intensity_max'])] = np.nan

    if sw_temp:
        for i in range(int(nBlocks)):
            tt = (year >= mhwBlock['years_start'][i]) * (year <= mhwBlock['years_end'][i])
            mhwBlock['temp_mean'][i] = np.nanmean(temp[tt])
            mhwBlock['temp_max'][i] = np.nanmax(temp[tt])
            mhwBlock['temp_min'][i] = np.nanmin(temp[tt])

    if removeMissing:
        missingYears = np.unique(year[np.where(clim['missing'])[0]])
        for y in range(len(missingYears)):
            iMissing = np.where((mhwBlock['years_start'] <= missingYears[y]) * (mhwBlock['years_end'] >= missingYears[y]))[0][0]
            mhwBlock['count'][iMissing] = np.nan
            mhwBlock['duration'][iMissing] = np.nan
            mhwBlock['intensity_max'][iMissing] = np.nan
            mhwBlock['intensity_max_max'][iMissing] = np.nan
            mhwBlock['intensity_mean'][iMissing] = np.nan
            mhwBlock['intensity_cumulative'][iMissing] = np.nan
            mhwBlock['intensity_var'][iMissing] = np.nan
            mhwBlock['intensity_max_relThresh'][iMissing] = np.nan
            mhwBlock['intensity_mean_relThresh'][iMissing] = np.nan
            mhwBlock['intensity_cumulative_relThresh'][iMissing] = np.nan
            mhwBlock['intensity_var_relThresh'][iMissing] = np.nan
            mhwBlock['intensity_max_abs'][iMissing] = np.nan
            mhwBlock['intensity_mean_abs'][iMissing] = np.nan
            mhwBlock['intensity_cumulative_abs'][iMissing] = np.nan
            mhwBlock['intensity_var_abs'][iMissing] = np.nan
            mhwBlock['total_days'][iMissing] = np.nan
            mhwBlock['total_icum'][iMissing] = np.nan

    return mhwBlock


def meanTrend(mhwBlock, alpha=0.05):
    '''

    Calculates the mean and trend of marine heatwave (MHW) properties. Takes as input a
    collection of block-averaged MHW properties (using the marineHeatWaves.blockAverage
    function). Handles missing values (which should be specified by NaNs).

    Inputs:

      mhwBlock      Time series of block-averaged MHW statistics calculated using the
                    marineHeatWaves.blockAverage function
      alpha         Significance level for estimate of confidence limits on trend, e.g.,
                    alpha = 0.05 for 5% significance (or 95% confidence) (DEFAULT = 0.05)

    Outputs:

      mean          Mean of all MHW properties over all block-averaged values
      trend         Linear trend of all MHW properties over all block-averaged values
      dtrend        One-sided width of (1-alpha)% confidence intevfal on linear trend,
                    i.e., trend lies within (trend-dtrend, trend+dtrend) with specified
                    level  of confidence.

                    Both mean and trend have the following keys, the units the trend
                    are the units of the property of interest per year:

        'duration'             Duration of MHW [days]
        'intensity_max'        Maximum (peak) intensity [deg. C]
        'intensity_mean'       Mean intensity [deg. C]
        'intensity_var'        Intensity variability [deg. C]
        'intensity_cumulative' Cumulative intensity [deg. C x days]


    Notes:

      This calculation performs a multiple linear regression of the form
        y ~ beta * X + eps
      where y is the MHW property of interest and X is a matrix of predictors. The first
      column of X is all ones to estimate the mean, the second column is the time vector
      which is taken as mhwBlock['years_centre'] and offset to be equal to zero at its
      mid-point.

    '''

    # Initialize mean and trend dictionaries
    mean = {}
    trend = {}
    dtrend = {}

    t = mhwBlock['years_centre']
    X = np.array([np.ones(t.shape), t-t.mean()]).T

    for key in mhwBlock.keys():
        if (key == 'years_centre') + (key == 'years_end') + (key == 'years_start'):
            continue

        y = mhwBlock[key]
        valid = ~np.isnan(y) # non-NaN indices

        if np.isinf(nonans(y).sum()):
            beta = [np.nan, np.nan]
        elif np.sum(~np.isnan(y)) > 0:
            beta = linalg.lstsq(X[valid,:], y[valid])[0]
        else:
            beta = [np.nan, np.nan]

        mean[key] = beta[0]
        trend[key] = beta[1]

        # Confidence limits on trend
        yhat = np.sum(beta*X, axis=1)
        t_stat = stats.t.isf(alpha/2, len(t[valid])-2)
        s = np.sqrt(np.sum((y[valid] - yhat[valid])**2) / (len(t[valid])-2))
        Sxx = np.sum(X[valid,1]**2) - (np.sum(X[valid,1])**2)/len(t[valid]) # np.var(X, axis=1)[1]
        dbeta1 = t_stat * s / np.sqrt(Sxx)
        dtrend[key] = dbeta1

    # Return mean, trend
    return mean, trend, dtrend


def runavg(ts, w):
    '''

    Performs a running average of an input time series using uniform window
    of width w. This function assumes that the input time series is periodic.

    Inputs:

      ts            Time series [1D numpy array]
      w             Integer length (must be odd) of running average window

    Outputs:

      ts_smooth     Smoothed time series

    '''
    # Original length of ts
    N = len(ts)
    # make ts three-fold periodic
    ts = np.append(ts, np.append(ts, ts))
    # smooth by convolution with a window of equal weights
    ts_smooth = np.convolve(ts, np.ones(w)/w, mode='same')
    # Only output central section, of length equal to the original length of ts
    ts = ts_smooth[N:2*N]

    return ts


def pad(data, maxPadLength=False):
    '''

    Linearly interpolate over missing data (NaNs) in a time series.

    Inputs:

      data	     Time series [1D numpy array]
      maxPadLength   Specifies the maximum length over which to interpolate,
                     i.e., any consecutive blocks of NaNs with length greater
                     than maxPadLength will be left as NaN. Set as an integer.
                     maxPadLength=False (default) interpolates over all NaNs.

    '''
    data_padded = data.copy()
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data_padded[bad_indexes] = interpolated
    if maxPadLength:
        blocks, n_blocks = ndimage.label(np.isnan(data))
        for bl in range(1, n_blocks+1):
            if (blocks==bl).sum() > maxPadLength:
                data_padded[blocks==bl] = np.nan

    return data_padded


def nonans(array):
    '''
    Return input array [1D numpy array] with
    all nan values removed
    '''
    return array[~np.isnan(array)]



