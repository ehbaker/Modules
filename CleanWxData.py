'''
  functions to clean weather station data
'''
import numpy as np
import pandas as pd
from time import mktime as mktime

def decimal_date_from_julian(JD):
    L= JD+68569
    N= 4*L/146097
    L= L-(146097*N+3)/4
    I= 4000*(L+1)/1461001
    L= L-1461*I/4+31
    J= 80*L/2447
    K= L-2447*J/80
    L= J/11
    J= J+2-12*L
    decimal_date= 100*(N-49)+I+L
    return(decimal_date)
    

def define_water_year(dat):
    '''
    input: dataframe with time index
    '''
    dat['year']=dat.index.year
    dat['month']=dat.index.month
    dat['water_year']=dat.index.year
    dat.loc[dat.month.isin([10, 11, 12]), 'water_year']=dat.loc[dat.month.isin([10, 11, 12]), 'water_year']+1
    return(dat)

    
def correct_undercatch_yang98(df, precip_column, phase_column, wind_speed_column, inplace=True):
    '''
       phase must have 3 l3vels; created by "define_precip_phase" function: snow, rain, mixed.
       precipdata=pandas series of data (mm)
       tempdata= pandas series of temp data (C)
       This uses the equations in Yang 1998 (followed in Liljedahl 2017, among others) for the Alter sheild.
       
       output is new column in dataframe, with name of input column followed by "_undercatcj_adj"
    '''
   #Save original dataframe before editing precip
    df_unaltered=df.copy()
    
    #define windspeed threshold after which relationship no longer applicable
    ws_max=7 #limit as this is the highest that gage relationship showed, in Yang 98 paper
    
    #Create new column for output (only use this for TROUBLESHOOTING; otherwise)
    if inplace==False:
        new_column=precip_column +"_undercatch_adj"
        df[new_column]=np.nan
    else:
        new_column = precip_column
    
    #Correct Rain
    wind_speed_duringrain=df[wind_speed_column][df.phase=='rain'] #save wind speed
    wind_speed_duringrain[wind_speed_duringrain>ws_max]=ws_max #reset wind speed to max that is part of the calibrated equations, if is over
    correction_factor_rain=100/np.exp(4.606 -0.041*wind_speed_duringrain**0.69) #calculate correction factor
    correction_factor_rain[correction_factor_rain>100]=np.nan
    rain_orig=df[precip_column][df.phase=='rain'] #pull out original rain values
    rain_corrected=rain_orig*correction_factor_rain #calculate corrected rain values
    df.loc[df.phase=='rain', new_column]=rain_corrected #store corrected values in the correct place
    
    #Correct Snow
    wind_speed_duringsnow=df[wind_speed_column][df.phase=='snow'] #save wind speed
    wind_speed_duringsnow[wind_speed_duringsnow>ws_max]=ws_max#reset wind speed to max that is part of the calibrated equations, if is over
    correction_factor_snow=100/np.exp(4.606 -0.036*wind_speed_duringsnow**1.75) #calculate correction factor
    correction_factor_snow[correction_factor_snow>100]=np.nan #limit the range of the adjustment to < 100x ()16 m/s)
    snow_orig=df[precip_column][df.phase=='snow'] #pull out original snow values
    snow_corrected=snow_orig*correction_factor_snow #calculate corrected snow values
    df.loc[df.phase=='snow', new_column]=snow_corrected #store corrected values in the correct place
    
    #Correct Mixed
    wind_speed_duringmixed=df[wind_speed_column][df.phase=='mixed'] #save wind speed
    wind_speed_duringmixed[wind_speed_duringmixed>ws_max]=ws_max #reset wind speed to max that is part of the calibrated equations, if is over
    correction_factor_mixed=100/(101.04 -5.62 *wind_speed_duringmixed) #calculate correction factor
    correction_factor_mixed[correction_factor_mixed>100]=np.nan
    mixed_orig=df[precip_column][df.phase=='mixed'] #pull out original mixed values
    mixed_corrected=mixed_orig*correction_factor_mixed #calculate corrected mixed values
    df.loc[df.phase=='mixed', new_column]=mixed_corrected #store corrected values in the correct place
    
    #For places where precip data is available, but no wind speed results in an adjusted wind speed of NAN, revert to originally recorded data
    df.loc[df[new_column].isnull(), new_column]=df_unaltered.loc[df[new_column].isnull(), precip_column]
    
    return(df)
        
    
def define_precip_phase(df_orig, temp_column):
    '''need to pass names of columns in as strings
       thresholdes defined based on McCabe and Wolock, 2010; as suggested by Harpold, 2017 review paper
       : "Rain or snow: hydrologic processes, observations, prediction, and research needs" '''
    df=df_orig.copy() 
    df['phase']=np.nan
    df.loc[df[temp_column]<-1, 'phase']="snow"
    df.loc[df[temp_column]>3, 'phase']="rain"
    df.loc[(df[temp_column]>=-1) & (df[temp_column]<=3), 'phase']='mixed'
    df.phase=df.phase.astype('category')
    return(df)
    
def define_precip_rate(wx_dat, precip_col_name, trace_cutoff, high_cutoff):
    '''
    dat: pandas dataframe containing data
    precip_col_name: name of pandas column with precip data (non-cumulative)
    trace_cutoff: all precip below this amount will be labeled a "trace"
    high_cutoff: all precip above this amount will be labeled "high"
    
    returns: dataframe with column 'precip-rate' that gives categorical variable of 3 precip rate categories, based on input
    '''
    #Create high, medium, and low precip phase ID
    wx_dat['precip_rate']=pd.np.nan
    wx_dat.loc[(wx_dat[precip_col_name]<trace_cutoff), 'precip_rate']='trace'
    wx_dat.loc[(wx_dat[precip_col_name]>high_cutoff), 'precip_rate']='heavy'
    wx_dat.loc[(wx_dat[precip_col_name]>trace_cutoff) & (wx_dat[precip_col_name]<high_cutoff), 'precip_rate']='medium'

    
    
def adjustforwettingloss(df, precip_column, inplace=True):
    '''
      input should be the ALREADY UNDERCATCH ADJUSTED output of correct_undercatch_yang1998() for STAGE GAGE
      returns dataframe with new data in column named "wetting_loss_adjusted"
      
      this function adjusts precipitation data for wetting loss following Yang 1998 and Liljedahl 2017.
      It adds 0.03 and 0.15mm respectively to measured daily rainfall and snowfall.
      function should be used AFTER application of yang1998 precip undercatch correction equations (on the output);
      wetting loss is added after correction, not as a part of the correction that will also be scaled by undercatch calcs.
    '''
    df=df.copy() #this line prevents the function from changing data in the input dataframe! (Python chained indexing issues)
    if inplace==False:
        new_name=precip_column+ "_wetting_loss_adjusted"
        df[new_name]=np.nan
    else:
        new_name = precip_column
    
    precip_orig_snow=df[precip_column][(df[precip_column]>0) & (df['phase']=='snow')].copy()
    df.loc[(df[precip_column]>0) & (df['phase']=='snow'), new_name]= precip_orig_snow + 0.15
    
    precip_orig_rain=df[precip_column][(df[precip_column]>0) & (df['phase']!='snow')].copy()
    df.loc[(df[precip_column]>0) & (df['phase']!='snow'), new_name]= precip_orig_rain + 0.03
    return(df)

def clean_wind_speed_data(dat, wind_col, max_cutoff=75):
    '''
    clean wind speed data; very specific to our sites at this point
    set wind speed to NAN where hourly variability is < 0.05 m/s
      - also here, must be 0, and before 2014/04 to be changed. More recnt data has fewer digits recorded, complicating this.
    '''
    dat.loc[dat[wind_col]>75, wind_col]=pd.np.nan #limit from WMO (Zahumensky, 2004) pub
    
    #Set values where instantaneous variability below minimum (rimed or other probelm) to NAN
    hourly_dat=pd.DataFrame()
    hourly_wind=dat[wind_col].resample('H')
    hourly_dat['ws_sum']=hourly_wind.sum()
    variability_under_WMO_cutoff=(hourly_wind.max()-hourly_wind.min())<0.5
    hourly_dat['wind_var']=variability_under_WMO_cutoff
    hourly_dat['is_zero']=hourly_dat.ws_sum==0
    hourly_dat['before_sensitivity_change']=hourly_dat.index<='2017-04' #with decrease in logger digits recorded, test becomes unreliable.

    #Set times where the conditions are met to NAN
    bool_idx_bad_days=hourly_dat.wind_var & hourly_dat.is_zero & hourly_dat.before_sensitivity_change
    fifteenmin_bool_idx=bool_idx_bad_days.resample('15min').ffill() #change back to 15 min from hourly
    fifteenmin_bool_idx[fifteenmin_bool_idx.isnull()]=False
    fifteenmin_df=pd.DataFrame(fifteenmin_bool_idx, columns=['set_wind_to_null'])
    
    dat2=dat.merge(fifteenmin_df, left_index=True, right_index=True, how='left') #add to same dataframe; hav
    dat2.loc[dat2.set_wind_to_null.isnull(), 'set_wind_to_null']=False #tail has remaining NULLs
    
    #Set locations to null
    dat.loc[dat2.set_wind_to_null, 'WindSpeed']=pd.np.nan
    
    return(dat)
    
def aggregate_time_with_threshold(ser, time_resample_code, func="mean"):    
    '''
    ser: pandas series
    func - can be sum or mean
    '''
    mth_dat=ser.resample(time_resample_code, convention='start').agg([func, 'count'])
    mth_dat.rename(columns={func:ser.name}, inplace=True)
    
    #define timestep-dependent things
    if time_resample_code in ['M', 'MS']:
        steps_in_period=mth_dat.index.days_in_month
    elif time_resample_code in ['A', 'Y', 'YS', 'AS']:
        steps_in_period=pd.Series([365.0]*len(mth_dat))
        steps_in_period.index=mth_dat.index
    else:
        print("stop! this freqency hasn't been prepared for; edit function")
   
    # determine invalid steps
   # mth_dat['n_allowed_missing']=0.1 * steps_in_period
    mth_dat['pct']=mth_dat['count']/steps_in_period
    invalid= mth_dat['count'] <= 0.9 * steps_in_period
    mth_dat['invalid']=invalid
    
    #Set invalid steps to NAN
    mth_dat.loc[mth_dat.invalid, ser.name]=pd.np.nan
    
    mth_dat.drop(['invalid', 'count'], axis=1, inplace=True)
    return(mth_dat)
    
    
#def toYearFraction(date):
#    def sinceEpoch(date): # returns seconds since epoch
#        return mktime(date.timetuple())
#    s = sinceEpoch
#
#    year = date.year
#    startOfThisYear = pd.datetime(year=year, month=1, day=1)
#    startOfNextYear = pd.datetime(year=year+1, month=1, day=1)
#
#    yearElapsed = s(date) - s(startOfThisYear)
#    yearDuration = s(startOfNextYear) - s(startOfThisYear)
#    fraction = yearElapsed/yearDuration
#
#    return (date.year + fraction)
#    
    
from datetime import datetime as dt
import time

def toYearFraction(date):
    '''
    '''
    def sinceEpoch(date): # returns seconds since epoch
        #return time.mktime(date.timetuple())
        epoch = dt(1970, 1, 1)
        diff=date- epoch
        return(diff.total_seconds())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction
    
    
    
    
    
    
    
    
    