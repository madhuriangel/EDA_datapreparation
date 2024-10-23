'''
 A set of functions which implement the Marine Heat Wave (MHW) definition of Hobday et al. (2016)

'''
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import stats
import scipy.ndimage as ndimage
from datetime import date

def detect(t, temp, climatologyPeriod=[None, None], pctile=90, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=False, alternateClimatology=False, Ly=False):
    '''
   Applies the Hobday et al. (2016) marine heat wave definition to an input time
   series of temp ('temp') along with a time vector ('t'). Outputs properties of
   all detected marine heat waves.

   Inputs:

     t       Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
             [1D numpy array of length T]
     temp    Temperature vector [1D numpy array of length T]

   Outputs:

     mhw     Detected marine heat waves (MHWs). Each key (following list) is a
             list of length N where N is the number of detected MHWs:

       'time_start'           Start time of MHW [ordinal format]
       'time_end'             End time of MHW [ordinal format]
       'time_peak'            Time of MHW peak [ordinal format]
       'date_start'           Start date of MHW [datetime format]
       'date_end'             End date of MHW [datetime format]
       'date_peak'            Date of MHW peak [datetime format]
       'index_start'          Start index of MHW
       'index_end'            End index of MHW
       'index_peak'           Index of MHW peak
       'duration'             Duration of MHW [days]
       'intensity_max'        Maximum (peak) intensity [deg. C]
       'intensity_mean'       Mean intensity [deg. C]
       'intensity_var'        Intensity variability [deg. C]
       'intensity_cumulative' Cumulative intensity [deg. C x days]
       'intensity_max_relThresh', 'intensity_mean_relThresh', 'intensity_var_relThresh', 
       and 'intensity_cumulative_relThresh' are as above except relative to the
       threshold (e.g., 90th percentile) rather than the seasonal climatology

       'intensity_max_abs', 'intensity_mean_abs', 'intensity_var_abs', and
       'intensity_cumulative_abs' are as above except as absolute magnitudes
       rather than relative to the seasonal climatology or threshold

       'n_events'             A scalar integer (not a list) indicating the total
                              number of detected MHW events

       clim    Climatology of SST. Each key (following list) is a seasonally-varying
             time series [1D numpy array of length T] of a particular measure:

       'thresh'               Seasonally varying threshold (e.g., 90th percentile)
       'seas'                 Climatological seasonal cycle
       'missing'              A vector of TRUE/FALSE indicating which elements in 
                              temp were missing values for the MHWs detection

   Notes:

     1. This function assumes that the input time series consist of continuous daily values
        with few missing values. Time ranges which start and end part-way through the calendar
        year are supported.

     2. This function supports leap years. This is done by ignoring Feb 29s for the initial
        calculation of the climatology and threshold. The value of these for Feb 29 is then
        linearly interpolated from the values for Feb 28 and Mar 1.

    '''
    # Initialize MHW output variable
    mhw = {}
    mhw['time_start'] = []
    mhw['time_end'] = []
    mhw['time_peak'] = []
    mhw['date_start'] = []
    mhw['date_end'] = []
    mhw['date_peak'] = []
    mhw['index_start'] = []
    mhw['index_end'] = []
    mhw['index_peak'] = []
    mhw['duration'] = []
    mhw['intensity_max'] = []
    mhw['intensity_mean'] = []
    mhw['intensity_var'] = []
    mhw['intensity_cumulative'] = []
    mhw['intensity_max_relThresh'] = []
    mhw['intensity_mean_relThresh'] = []
    mhw['intensity_var_relThresh'] = []
    mhw['intensity_cumulative_relThresh'] = []
    mhw['intensity_max_abs'] = []
    mhw['intensity_mean_abs'] = []
    mhw['intensity_var_abs'] = []
    mhw['intensity_cumulative_abs'] = []
    
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

    year_leapYear = 2012
    t_leapYear = np.arange(date(year_leapYear, 1, 1).toordinal(), date(year_leapYear, 12, 31).toordinal() + 1)
    dates_leapYear = [date.fromordinal(tt.astype(int)) for tt in t_leapYear]
    month_leapYear = np.zeros((len(t_leapYear)))
    day_leapYear = np.zeros((len(t_leapYear)))
    doy_leapYear = np.zeros((len(t_leapYear)))
    for tt in range(len(t_leapYear)):
        month_leapYear[tt] = date.fromordinal(t_leapYear[tt]).month
        day_leapYear[tt] = date.fromordinal(t_leapYear[tt]).day
        doy_leapYear[tt] = t_leapYear[tt] - date(date.fromordinal(t_leapYear[tt]).year, 1, 1).toordinal() + 1

    for tt in range(T):
        doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])]

    feb28 = 59
    feb29 = 60

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

    if maxPadLength:
        temp = pad(temp, maxPadLength=maxPadLength)
        tempClim = pad(tempClim, maxPadLength=maxPadLength)

    lenClimYear = 366
    clim_start = np.where(yearClim == climatologyPeriod[0])[0][0]
    clim_end = np.where(yearClim == climatologyPeriod[1])[0][-1]
    thresh_climYear = np.NaN * np.zeros(lenClimYear)
    seas_climYear = np.NaN * np.zeros(lenClimYear)
    clim = {}
    clim['thresh'] = np.NaN * np.zeros(TClim)
    clim['seas'] = np.NaN * np.zeros(TClim)

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

    thresh_climYear[feb29 - 1] = 0.5 * thresh_climYear[feb29 - 2] + 0.5 * thresh_climYear[feb29]
    seas_climYear[feb29 - 1] = 0.5 * seas_climYear[feb29 - 2] + 0.5 * seas_climYear[feb29]

    if smoothPercentile:
        if Ly:
            valid = ~np.isnan(thresh_climYear)
            thresh_climYear[valid] = runavg(thresh_climYear[valid], smoothPercentileWidth)
            valid = ~np.isnan(seas_climYear)
            seas_climYear[valid] = runavg(seas_climYear[valid], smoothPercentileWidth)
        else:
            thresh_climYear = runavg(thresh_climYear, smoothPercentileWidth)
            seas_climYear = runavg(seas_climYear, smoothPercentileWidth)

    clim['thresh'] = thresh_climYear[doy.astype(int) - 1]
    clim['seas'] = seas_climYear[doy.astype(int) - 1]

    clim['missing'] = np.isnan(temp)
    temp[np.isnan(temp)] = clim['seas'][np.isnan(temp)]

    exceed_bool = temp - clim['thresh']
    exceed_bool[exceed_bool <= 0] = False
    exceed_bool[exceed_bool > 0] = True
    exceed_bool[np.isnan(exceed_bool)] = False
    events, n_events = ndimage.label(exceed_bool)
    

    for ev in range(1, n_events + 1):
        event_duration = (events == ev).sum()
        if event_duration < minDuration:
            continue
        mhw['time_start'].append(t[np.where(events == ev)[0][0]])
        mhw['time_end'].append(t[np.where(events == ev)[0][-1]])

    if joinAcrossGaps:
        gaps = np.array(mhw['time_start'][1:]) - np.array(mhw['time_end'][0:-1]) - 1
        if len(gaps) > 0:
            while gaps.min() <= maxGap:
                ev = np.where(gaps <= maxGap)[0][0]
                mhw['time_end'][ev] = mhw['time_end'][ev + 1]
                del mhw['time_start'][ev + 1]
                del mhw['time_end'][ev + 1]
                gaps = np.array(mhw['time_start'][1:]) - np.array(mhw['time_end'][0:-1]) - 1
                if len(gaps) == 0:
                    break

    mhw['n_events'] = len(mhw['time_start'])

    for ev in range(mhw['n_events']):
        mhw['date_start'].append(date.fromordinal(mhw['time_start'][ev]))
        mhw['date_end'].append(date.fromordinal(mhw['time_end'][ev]))
        tt_start = np.where(t == mhw['time_start'][ev])[0][0]
        tt_end = np.where(t == mhw['time_end'][ev])[0][0]
        mhw['index_start'].append(tt_start)
        mhw['index_end'].append(tt_end)
        temp_mhw = temp[tt_start:tt_end + 1]
        thresh_mhw = clim['thresh'][tt_start:tt_end + 1]
        seas_mhw = clim['seas'][tt_start:tt_end + 1]
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
    '''

    Calculate statistics of marine heatwave (MHW) properties averaged over blocks of
    a specified length of time. Takes as input a collection of detected MHWs
    (using the mhw_hobday.detect function) and a time vector for the source
    SST series.

    Inputs:

      t       Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
      mhw     Marine heat waves (MHWs) detected using marineHeatWaves.detect

    Outputs:

      mhwBlock   Time series of block-averaged MHW properties. Each key (following list)
                 is a list of length N where N is the number of blocks:
 
        'years_start'          Start year blocks (inclusive)
        'years_end'            End year of blocks (inclusive)
        'years_centre'         Decimal year at centre of blocks
        'count'                Total MHW count in each block
        'duration'             Average MHW duration in each block [days]
        'intensity_max'        Average MHW "maximum (peak) intensity" in each block [deg. C]
        'intensity_max_max'    Maximum MHW "maximum (peak) intensity" in each block [deg. C]
        'intensity_mean'       Average MHW "mean intensity" in each block [deg. C]
        'intensity_var'        Average MHW "intensity variability" in each block [deg. C]
        'intensity_cumulative' Average MHW "cumulative intensity" in each block [deg. C x days]
        'total_days'           Total number of MHW days in each block [days]
        'total_icum'           Total cumulative intensity over all MHWs in each block [deg. C x days]

        'intensity_max_relThresh', 'intensity_mean_relThresh', 'intensity_var_relThresh', 
        and 'intensity_cumulative_relThresh' are as above except relative to the
        threshold (e.g., 90th percentile) rather than the seasonal climatology

        'intensity_max_abs', 'intensity_mean_abs', 'intensity_var_abs', and
        'intensity_cumulative_abs' are as above except as absolute magnitudes
        rather than relative to the seasonal climatology or threshold

    Options:

      blockLength            Size of block (in years) over which to calculate the
                             averaged MHW properties. Must be an integer greater than
                             or equal to 1 (DEFAULT = 1 [year])
      removeMissing          Boolean switch indicating whether to remove (set = NaN)
                             statistics for any blocks in which there were missing 
                             temperature values (DEFAULT = FALSE)
      clim                   The temperature climatology (including missing value information)
                             as output by mhw_hobday.detect (required if removeMissing = TRUE)
      temp                   Temperature time series. If included mhwBlock will output block
                             averages of mean, max, and min temperature (DEFAULT = NONE)

    '''

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
    # Replace empty years in intensity_max_max
    mhwBlock['intensity_max_max'][np.isnan(mhwBlock['intensity_max'])] = np.nan

    # Temperature series
    if sw_temp:
        for i in range(int(nBlocks)):
            tt = (year >= mhwBlock['years_start'][i]) * (year <= mhwBlock['years_end'][i])
            mhwBlock['temp_mean'][i] = np.nanmean(temp[tt])
            mhwBlock['temp_max'][i] = np.nanmax(temp[tt])
            mhwBlock['temp_min'][i] = np.nanmin(temp[tt])

    #
    # Remove years with missing values
    #

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
                    mhw_hobday.blockAverage function
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

        'intensity_max_relThresh', 'intensity_mean_relThresh', 'intensity_var_relThresh', 
        and 'intensity_cumulative_relThresh' are as above except relative to the
        threshold (e.g., 90th percentile) rather than the seasonal climatology

        'intensity_max_abs', 'intensity_mean_abs', 'intensity_var_abs', and
        'intensity_cumulative_abs' are as above except as absolute magnitudes
        rather than relative to the seasonal climatology or threshold

    Notes:

      This calculation performs a multiple linear regression of the form
        y ~ beta * X + eps
      where y is the MHW property of interest and X is a matrix of predictors. The first
      column of X is all ones to estimate the mean, the second column is the time vector
      which is taken as mhwBlock['years_centre'] and offset to be equal to zero at its
      mid-point.

    Reusing code, Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb-Mar 2015

    '''

    # Initialize mean and trend dictionaries
    mean = {}
    trend = {}
    dtrend = {}

    # Construct matrix of predictors, first column is all ones to estimate the mean,
    # second column is the time vector, equal to zero at mid-point.
    t = mhwBlock['years_centre']
    X = np.array([np.ones(t.shape), t-t.mean()]).T

    # Loop over all keys in mhwBlock
    for key in mhwBlock.keys():
        # Skip time-vector keys of mhwBlock
        if (key == 'years_centre') + (key == 'years_end') + (key == 'years_start'):
            continue

        # Predictand (MHW property of interest)
        y = mhwBlock[key]
        valid = ~np.isnan(y) # non-NaN indices

        # Perform linear regression over valid indices
        if np.isinf(nonans(y).sum()): # If contains Inf values
            beta = [np.nan, np.nan]
        elif np.sum(~np.isnan(y)) > 0: # If at least one non-NaN value
            beta = linalg.lstsq(X[valid,:], y[valid])[0]
        else:
            beta = [np.nan, np.nan]

        # Insert regression coefficients into mean and trend dictionaries
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


def rank(t, mhw):
    '''

    Calculate the rank and return periods of marine heatwaves (MHWs) according to
    each metric. Takes as input a collection of detected MHWs (using the
    mhw_hobday.detect function) and a time vector for the source SST series.

    Inputs:

      t       Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
      mhw     Marine heat waves (MHWs) detected using mhw_hobday.detect

    Outputs:

      rank          The rank of each MHW according to each MHW property. A rank of 1 is the
                    largest, 2 is the 2nd largest, etc. Each key (listed below) is a list
                    of length N where N is the number of MHWs.

      returnPeriod  The return period (in years) of each MHW according to each MHW property.
                    The return period signifies, statistically, the recurrence interval for
                    an event at least as large/long as the event in quetion. Each key (listed
                    below) is a list of length N where N is the number of MHWs.
 
        'duration'             Average MHW duration in each block [days]
        'intensity_max'        Average MHW "maximum (peak) intensity" in each block [deg. C]
        'intensity_mean'       Average MHW "mean intensity" in each block [deg. C]
        'intensity_var'        Average MHW "intensity variability" in each block [deg. C]
        'intensity_cumulative' Average MHW "cumulative intensity" in each block [deg. C x days]
        'total_days'           Total number of MHW days in each block [days]
        'total_icum'           Total cumulative intensity over all MHWs in each block [deg. C x days]

        'intensity_max_relThresh', 'intensity_mean_relThresh', 'intensity_var_relThresh', 
        and 'intensity_cumulative_relThresh' are as above except relative to the
        threshold (e.g., 90th percentile) rather than the seasonal climatology

        'intensity_max_abs', 'intensity_mean_abs', 'intensity_var_abs', and
        'intensity_cumulative_abs' are as above except as absolute magnitudes
        rather than relative to the seasonal climatology or threshold

    Notes:

      This function assumes that the MHWs were calculated over a suitably long record that return
      periods make sense. If the record length is a few years or less than this becomes meaningless.

   
    '''

    # Initialize rank and return period dictionaries
    rank = {}
    returnPeriod = {}

    # Number of years on record
    nYears = len(t)/365.25

    # Loop over all keys in mhw
    for key in mhw.keys():
        # Skip irrelevant keys of mhw, only calculate rank/returns for MHW properties
        if (key == 'date_end') + (key == 'date_peak') + (key == 'date_start') + (key == 'date_end') + (key == 'date_peak') + (key == 'date_start') + (key == 'index_end') + (key == 'index_peak') + (key == 'index_start') + (key == 'n_events'):
            continue

        # Calculate ranks
        rank[key] = mhw['n_events'] - np.array(mhw[key]).argsort().argsort()  
        # Calculate return period as (# years on record + 1) / (# of occurrences of event)
        # Return period is for events of at least the event magnitude/duration
        returnPeriod[key] = (nYears + 1) / rank[key]

    # Return rank, return
    return rank, returnPeriod


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
