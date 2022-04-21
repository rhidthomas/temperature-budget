import numpy as np
import xarray as xr
#import matplotlib.pyplot as plt
from datetime import datetime
import seager_budget_functions as sbf
           
#%% To-do list  

# Account for topology: ps
# Save as individual NetCDFs, not as all_seas dict
# Change var names, e.g. stw --> st_ed?
# Contour plot smoothing

glob_attrs = {'creation_date':str(datetime.now()), 
         'author':'Rhidian Thomas', 
         'email':'rhidian.thomas@physics.ox.ac.uk',
         'units':'K/s'}
      
# Time-mean is mean of that season, in that year; prime = daily deviation
                
g = 9.81 #m/s^2  

base_data_dir = "/network/aopp/hera/mad/rhidian/data/ERA5/dailymean"
save_dir = "/network/aopp/hera/mad/rhidian/data/ERA5/temperature_budget"                
                

seasonal_slices = {'MAM':('{yr}-03-01','{yr}-05-31'),
                   'JJA':('{yr}-06-01','{yr}-08-31'),
                   'SON':('{yr}-09-01','{yr}-11-30'),
                   'DJF':('{yr}-12-01','{next_yr}-02-28')}

seasons = ['DJF','MAM','JJA','SON']
varlist = ['va','ta','wap']

plevs = [1000,925,850,750,700,650,600,550,500,450,
         400,350,300,250,200,150,100,70,50,10]

first_year = 1979
last_year  = 2020

year_range   = range(1979,2021)

# Return zonal-mean or not
zm = False


###### Masking syntax   #########
# mask = da.isin(da.sel(level=gl_levels).sel(latitude=gl_lats).sel(longitude=gl_lons))
# ^ isin returns True where Greenland is located
# da_masked = da.where(~mask)
# ^ ~ here flips True -> False in mask; da.where will return only True regions and mask
# out False regions as nan

### Greenland coords:
gl_lats = slice(84, 60)
gl_lons = slice(-74+360, -11+360)
gl_levels = slice(1000, 640)
###
#################################


# All budget results will be stored in this dict
all_seas = {seas : {} for seas in seasons}


for seas in seasons:
    # dailymean files contain all months in all years
    # Select one season at a time to analyse
    
    # Store budget terms for each year, then concat into an xarray after loop  

    start_dates = [] # Use this to label entries in xarray
    MMC_list = []
    stw_list = []
    tr_ed_list = []
    dT_list = []
    dynamics_list = []
    Q_list = []
    

    
    for year in year_range:
        # for DJF, this is year that December falls in
        # e.g. seas=DJF, yr=1980 --> December1980-Feb1981        
        
        # dailymean files contain all months in all years
        # Select one season in one year at a time to analyse
        print("[INFO] Processing {seas} {year}".format(
                seas=seas, year=year))
        
        start_date = seasonal_slices[seas][0].format(yr=year)
        if seas == 'DJF':
            # DJF wraps around end of calendar year
            final_date = seasonal_slices[seas][1].format(next_yr=year+1)
        else:
            final_date = seasonal_slices[seas][1].format(yr=year)
            
        full_seas_slice = slice(start_date,final_date)
        
        # dict to store array for each var, at each level
        da_store = {v: {} for v in varlist}
        
        for var in varlist:
            for plev in plevs:

                full_data_dir = "{bdir}/{var}/".format(
                        bdir=base_data_dir, var=var)
                
                filename = "{fdir}ERA5_dailymean_{var}{plev}_allYear_{fy}_{ly}.nc".format(
                            fdir=full_data_dir, var=var, plev=plev, fy=first_year, ly=last_year)
                ds = xr.open_dataset(filename)
                
                # Select only this season
                ds = ds.sel(time=full_seas_slice)
                
                da_store[var][plev] = ds[var]   
        
        # dict to store the full array (containing all levels) for each
        # variable, its time-mean ("time_mean") over the season, and deviation
        # from the time-mean ("prime")
        var_store = {v: {} for v in varlist}
    
        # Time means and deviations for this season:
        for var in varlist:
            # Need a list of arrays (one array per plevel) in order to concat 
            # into single array containing all levels
            datasets = [da_store[var][p] for p in plevs]
            
            var_store[var][var] = xr.concat(datasets, dim='level')
            da_time_mean = var_store[var][var].mean(dim='time')
            
            ### GREENLAND MASKING: COMMENT TO REMOVE MASKING
            
            #print('masking: {var}'.format(var=var))
            #da = xr.concat(datasets, dim='level')
            #mask = da.isin(da.sel(level=gl_levels).sel(latitude=gl_lats).sel(longitude=gl_lons))
            #var_store[var][var] = da.where(~mask)
            
            ### END MASKING ###
            
            # new masking strat: only time mean!
            #mask = da_time_mean.isin(da_time_mean.sel(level=gl_levels).sel(latitude=gl_lats).sel(longitude=gl_lons))
            #var_store[var]['time_mean'] = da_time_mean.where(~mask)
            
            var_store[var]['time_mean'] = da_time_mean
            var_store[var]['prime'] = var_store[var][var] - var_store[var]['time_mean']

        
        # This *does* kind of make the above loop a bit pointless...remove it later!!#
        va_time_mean = var_store['va']['time_mean']
        ta_time_mean = var_store['ta']['time_mean']
        wap_time_mean = var_store['wap']['time_mean']
        
        va_prime = var_store['va']['prime']
        ta_prime = var_store['ta']['prime']
        wap_prime = var_store['wap']['prime']
        

        # LHS of budget is \bar{d<T>/dt}, where \bar{} denotes seasonal mean
        # => \bar{d<T>/dt} = temperature difference b/w start & end of season
        # Units are effectively K/seas
        
        dT = (var_store['ta']['ta'].isel(time=-1) - 
                   var_store['ta']['ta'].isel(time=0)).mean('longitude')
        
        # RHS terms will be calculated below in units of K/s
        # Multiply by number of seconds in season to get units of K/seas
        days_in_seas = len(var_store['ta']['ta'].time)
        seas_length  = days_in_seas*24*3600 # number of seconds in season
        

        # Calculate the terms: each returned as xr.Dataset, with variable
        # for each component (meridional, vertical, etc)
        
        print('[INFO] Calculating budget terms')
        MMC = sbf.temperature_budget_MMC(va_time_mean=va_time_mean,
                                         ta_time_mean=ta_time_mean,
                                         wap_time_mean=wap_time_mean)
        
        stw = sbf.temperature_budget_stat_waves(va_time_mean=va_time_mean,
                                                ta_time_mean=ta_time_mean,
                                                wap_time_mean=wap_time_mean,
                                                zm=False)
        
        tr_ed = sbf.temperature_budget_transient_eddies(va_prime=va_prime,
                                                        ta_prime=ta_prime,
                                                        wap_prime=wap_prime,
                                                        zm=False)
        
        
        
        # Convert units to K/season
        terms = [MMC, stw, tr_ed]
        
        for i in range(len(terms)):
            terms[i] *= seas_length
            terms[i].attrs['units'] = 'K/season'
        
        # Use this for the other terms...
        glob_attrs['units'] = 'K/season'
        
        # Sum of dynamical terms (always a zonal-mean quantity)
        if zm:
            dynamics = MMC.total + stw.total + tr_ed.total
        else:
            dynamics = MMC.total + stw.total.mean('longitude') + tr_ed.total.mean('longitude')
        
        
        # Evaluate diabatic heating (Q) as budget residual
        Q = dT - dynamics
               
        # Store in lists for concatenation later
        MMC_list.append(MMC)
        stw_list.append(stw)
        tr_ed_list.append(tr_ed)
        
        dT_list.append(dT)
        Q_list.append(Q)
        dynamics_list.append(dynamics)
        
        start_dates.append(np.datetime64(start_date))        
        
    # Have now processed all years for this season! 
    print('[INFO] Finished all years for {seas}'.format(seas=seas))
    
    
    # Now concatenate lists into Datasets for whole timeseries
    dates = xr.DataArray(data=start_dates,dims='time',coords={'time':start_dates})
    
    # To loop through
    dict_of_lists = {'MMC':MMC_list,'stw':stw_list,'tr_ed':tr_ed_list,
                     'dT':dT_list,'Q':Q_list,'dynamics':dynamics_list}
    
    # Dummy savefile
    base_savefile = '{sdir}/{term_name}_tendency_zonmean_{zm}_{seas}_{fy}_{ly}.nc'.format(
            sdir=save_dir, fy=first_year, ly=last_year, zm=zm, seas=seas, term_name='{term_name}')
    
    
    # Assign attrs for remaining terms
    long_names = {'dynamics':'total dynamics', 
                'Q':'diabatic heating', 
                'dT':'temperature tendency'}
    
    # Conncat and save each term
    for term_name in dict_of_lists.keys():
            
        full_savefile = base_savefile.format(term_name=term_name)
        
        # Concatenate all years into single dataset
        temp = xr.concat(dict_of_lists[term_name],dim=dates)
        
        # Assigning attrs for those terms without them 
        if term_name in long_names.keys():
            temp = temp.assign_attrs(glob_attrs)
            temp.attrs['long_name'] = long_names[term_name]
        
        # May have to restart kernel...
        temp.to_netcdf(full_savefile)
        
print('[INFO] FINISHED WITHOUT AN ERROR!')    
