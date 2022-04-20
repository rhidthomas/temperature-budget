import numpy as np 

# Constants
a = 6400e3 # Earth radius [m]
R = 8.31 # molar gas constant [J/K/mol]
cp = 0.02914e3 # molar specific heat [J/kg/mol]

# REMEMEBR P UNITS ARE HPA NOT PA - FACTOR OF 100 DIFFERENCE!

def temperature_budget_MMC_meridional_advection(va_time_mean_zon_mean, ta_time_mean_zon_mean):
    # First term in MMC group of Seager temperature budget
    # <v_bar>/a * d<T_bar>/d_phi
    
    # 180/pi converts d/d_phi in degrees (default) to radians
    result = -(va_time_mean_zon_mean/a)*(180/np.pi)*(ta_time_mean_zon_mean.differentiate('latitude'))
    
    return result.transpose('level','latitude')

def temperature_budget_MMC_vertical_advection(wap_time_mean_zon_mean, ta_time_mean_zon_mean):
    # Second term in MMC group of Seager temperature budget
    # <omega_bar> * d<T_bar>/d_p
    
    result = -wap_time_mean_zon_mean*(ta_time_mean_zon_mean.differentiate('level'))
    # Convert hPa to Pa: divide by 100
    result = result/100
    
    return result.transpose('level','latitude')

def temperature_budget_MMC_adiabatic(wap_time_mean_zon_mean, ta_time_mean_zon_mean):
    # Third term in MMC group of Seager temperature budget
    # -R/(p*c_p) * <omega_bar>*<T_bar>
    
    
    p = wap_time_mean_zon_mean.level
    
    result = (R/(p*cp)) * wap_time_mean_zon_mean*ta_time_mean_zon_mean
    # Convert hPa to Pa: divide by 100
    result = result/100    
    
    return result.transpose('level','latitude')
    



def temperature_budget_MMC(va_time_mean, ta_time_mean, wap_time_mean,return_components=False):

                                                     
    # Calculate zonal means
    va_time_mean_zon_mean  = va_time_mean.mean('longitude')
    ta_time_mean_zon_mean  = ta_time_mean.mean('longitude')
    wap_time_mean_zon_mean = wap_time_mean.mean('longitude')


    term1 = temperature_budget_MMC_meridional_advection(va_time_mean_zon_mean=va_time_mean_zon_mean,
                                                        ta_time_mean_zon_mean=ta_time_mean_zon_mean)

    term2 = temperature_budget_MMC_vertical_advection(wap_time_mean_zon_mean=wap_time_mean_zon_mean,
                                                      ta_time_mean_zon_mean=ta_time_mean_zon_mean)

    term3 = temperature_budget_MMC_adiabatic(wap_time_mean_zon_mean=wap_time_mean_zon_mean,
                                             ta_time_mean_zon_mean=ta_time_mean_zon_mean)                                               
                                  
    # Add up the terms:
    total = term1 + term2 + term3
    
    if return_components:
        result_dict = {'total' : total, 
                       'meridional' : term1, 
                       'vertical' : term2, 
                       'adiabatic' : term3}
    else:
        result_dict = {'total ': total}

    return result_dict

                                                

#####################
#%% Stationary waves
    
def temperature_budget_stat_waves_meridional(va_time_mean_star_ta_time_mean_star, zm):
    # First term in stationary waves group of Seager temperature budget
    # 1/acos(phi) * d/d_phi( ( <v*_bar T*_bar> ) cos(phi) )
    
    # Work out where to put this conversion eventually...
    lats = np.deg2rad(va_time_mean_star_ta_time_mean_star.latitude)
    coslats = np.cos(lats)
    
    result = -(1/(a*coslats))*(180/np.pi)*(
            (
            coslats*va_time_mean_star_ta_time_mean_star
            ).differentiate('latitude')
            )
    
    # Calculates zonal mean if specified and reorders dims
    return check_result_shape(result,zm)


def temperature_budget_stat_waves_vertical(wap_time_mean_star_ta_time_mean_star, zm):
    # Second term in stationary waves group of Seager temperature budget
    # - d/d_p( <omega*_bar T*_bar> )
    
    result = -(wap_time_mean_star_ta_time_mean_star).differentiate('level')
    # Convert hPa to Pa: divide by 100
    result = result/100    
    
    # Calculates zonal mean if specified and reorders dims
    return check_result_shape(result,zm)


def temperature_budget_stat_waves_adiabatic(wap_time_mean_star_ta_time_mean_star, zm):
    # Third term in stationary waves group of Seager temperature budget
    # R/(p*c_p) *  <omega*_bar T*_bar> 
    
    p = wap_time_mean_star_ta_time_mean_star.level
    
    result = (R/(p*cp))*wap_time_mean_star_ta_time_mean_star
    # Convert hPa to Pa: divide by 100
    result = result/100
            
    # Calculates zonal mean if specified and reorders dims
    return check_result_shape(result,zm)




def temperature_budget_stat_waves(va_time_mean, ta_time_mean, wap_time_mean, return_components=False, zm=True):
    

    # Calculate zonal means
    va_time_mean_zon_mean  = va_time_mean.mean('longitude')
    ta_time_mean_zon_mean  = ta_time_mean.mean('longitude')
    wap_time_mean_zon_mean = wap_time_mean.mean('longitude')

    # Calculate zonal deviations
    va_time_mean_star = va_time_mean - va_time_mean_zon_mean
    ta_time_mean_star = ta_time_mean - ta_time_mean_zon_mean
    wap_time_mean_star = wap_time_mean - wap_time_mean_zon_mean

    # Flux terms:
    va_time_mean_star_ta_time_mean_star   = va_time_mean_star*ta_time_mean_star
    wap_time_mean_star_ta_time_mean_star  = wap_time_mean_star*ta_time_mean_star

    term1 = temperature_budget_stat_waves_meridional(va_time_mean_star_ta_time_mean_star=va_time_mean_star_ta_time_mean_star,
                                                 zm=zm)
    
    term2 = temperature_budget_stat_waves_vertical(wap_time_mean_star_ta_time_mean_star=wap_time_mean_star_ta_time_mean_star,
                                                 zm=zm)
    
    term3 = temperature_budget_stat_waves_adiabatic(wap_time_mean_star_ta_time_mean_star=wap_time_mean_star_ta_time_mean_star,
                                                 zm=zm)
    
    # Add up the terms:
    total = term1 + term2 + term3
    
    
    if return_components:
        result_dict = {'total' : total, 
                       'meridional' : term1, 
                       'vertical' : term2, 
                       'adiabatic' : term3}
    else:
        result_dict = {'total ': total}    
    
    return result_dict



        
#%% Transient eddies
# NOTE: 25/03/2022: these have now absorbed "zonal mean eddies" - see 
# Peixoto & Oort p.63 Eq 4.9
        
def temperature_budget_transient_eddies_meridional(va_prime_ta_prime_time_mean, zm):
    # First term in transient eddies group of Seager temperature budget
    # 1/acos(phi) * d/d_phi( \bar( <v'T'> ) cos(phi) )
    
    # Work out where to put this conversion eventually...
    lats = np.deg2rad(va_prime_ta_prime_time_mean.latitude)
    coslats = np.cos(lats)
    
    result = -(1/(a*coslats))*(180/np.pi)*(
            (
            coslats*va_prime_ta_prime_time_mean
            ).differentiate('latitude')
            )
            
    # Calculates zonal mean if specified and reorders dims
    return check_result_shape(result,zm)


def temperature_budget_transient_eddies_vertical(wap_prime_ta_prime_time_mean, zm):
    # Second term in transient eddies group of Seager temperature budget
    # - d/d_p( \bar( <omega' T'> ) )
    
    result = -(wap_prime_ta_prime_time_mean).differentiate('level')
    # Convert hPa to Pa: divide by 100
    result = result/100
            
    # Calculates zonal mean if specified and reorders dims
    return check_result_shape(result,zm)

def temperature_budget_transient_eddies_adiabatic(wap_prime_ta_prime_time_mean, zm):
    # Third term in transient eddies group of Seager temperature budget
    # R/(p*c_p) * \bar( <omega*' T*'> ) 
    
    p = wap_prime_ta_prime_time_mean.level
    
    result = (R/(p*cp))*wap_prime_ta_prime_time_mean
    # Convert hPa to Pa: divide by 100
    result = result/100
            
    # Calculates zonal mean if specified and reorders dims
    return check_result_shape(result,zm)




def temperature_budget_transient_eddies(va_prime, ta_prime, wap_prime, return_components=False, zm=True):

                                                                
    lats = np.deg2rad(va_prime.latitude)
    p = va_prime.level


    ##### save some time later #####
    va_prime_ta_prime_time_mean  = (va_prime*ta_prime).mean('time')
    wap_prime_ta_prime_time_mean = (wap_prime*ta_prime).mean('time')
    ################################

    term1 = temperature_budget_transient_eddies_meridional(va_prime_ta_prime_time_mean=va_prime_ta_prime_time_mean,
                                                 zm=zm)
    term2 = temperature_budget_transient_eddies_vertical(wap_prime_ta_prime_time_mean=wap_prime_ta_prime_time_mean,
                                                 zm=zm)
    term3 = temperature_budget_transient_eddies_adiabatic(wap_prime_ta_prime_time_mean=wap_prime_ta_prime_time_mean,
                                                 zm=zm)

    # Add up the terms:
    total = term1 + term2 + term3
    
    if return_components:
        result_dict = {'total' : total, 
                       'meridional' : term1, 
                       'vertical' : term2, 
                       'adiabatic' : term3}
    else:
        result_dict = {'total ': total}    
    
    return result_dict                             
                                                                


def check_result_shape(result,zm):
    """
    Dimension ordering can be lost in calculation of terms - enforce it here:
       Reshapes DataArray to be: (plev,lat,lon) if zm=False
       OR takes zonal mean and reorders to get (plev, lat) if zm=True (zonal-mean)
    """
    
    result = result.transpose('level','latitude','longitude')
    
    if zm: # Take zonal mean
        result = result.mean('longitude')
        result = result.transpose('level','latitude')
            
    return result


# Draw grid for unevenly spaced levels
def edges(yr,xr=None):
    # Values are centred, except for first and last entries   
    levels = np.zeros(len(yr)+1)
    levels[0], levels[-1] = yr[0], yr[-1]

    # Find midpoint between levels
    deltas = np.array([yr[i]-yr[i-1] for i in range(1,len(yr))])
    midlevs = yr[:-1] + deltas/2
    
    levels[1:-1] = midlevs
    
    return levels
    
