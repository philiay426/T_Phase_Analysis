################created by Wenbo WU, at Caltech########
#########################02/05/2019##############################


import global_parameters_TOT as global_var
import warnings
import obspy
from obspy import read, UTCDateTime
import numpy as np
import obspy.signal.filter as obssig
import scipy.signal.signaltools as sigtool
from obspy.geodetics.base import  gps2dist_azimuth
import para_result_analysis
import re
import datetime
import copy
import sys

def read_trace_list(trace_list_file):
    """
    Read the trace names listed in 'trace_list_file'
    """
    trace_list = []
    with open(trace_list_file) as f:
        content = f.read().splitlines()
    return content


def find_time_relative_to_b(st,ref_time):
    """
    Compute the time separation between 'ref_time' and the begin time 'stats.sac.b'###
    """

    if ref_time.lower()!='b' and ref_time.lower()!='o' and ref_time.lower()[0]!='t':
        sys.exit('Wrong reference time! It must be b, o or t0-9!')

    dif_time=st.stats.sac[ref_time.lower()]-st.stats.sac.b


    return dif_time


def check_hypo_depth_dif(template_sac,target_sac):

#iniialize spatially_close with True
    spatially_close=True

#hypocenter difference between the paired events
    if global_var.gcarc_spherical:
        template_lat=template_sac.stats.sac.evla
        template_lon=template_sac.stats.sac.evlo
        target_lat  =target_sac.stats.sac.evla
        target_lon  =target_sac.stats.sac.evlo
        [distance_meter,az,baz]=gps2dist_azimuth(template_lat,template_lon,target_lat, \
                    target_lon,a=global_var.R_Eearth, f=0.0)
    else:
        [distance_meter,az,baz]=gps2dist_azimuth(template_lat,template_lon,target_lat, \
                    target_lon)
    dif_hypo_km=distance_meter/1.e3

#depth difference 
#make sure that the unit of depth in sac head is km!!!!!!!
    dif_depth_km=abs(template_sac.stats.sac.evdp-target_sac.stats.sac.evdp)

#if the differences are larger than the given thresholds, set 'spatially_close' to be False
    if(dif_hypo_km>global_var.dif_hypo_threshold_km or \
       dif_depth_km>global_var.dif_dep_threshold_km):
        spatially_close=False

    #print dif_hypo_km,dif_depth_km
    return spatially_close

def read_merged_CC_table(file_merged_CC_table,nstation):
    "read the merged CC table containing all the CC results from every station and event"
    with open(file_merged_CC_table) as f:
        content_read = f.read().splitlines()
    #start with content_read[1:] to skip the first line of head information

    merged_CC_table_read={}
    CC_one_sta=copy.deepcopy(para_result_analysis.CC_one_sta)
    for CC_this_pair in content_read[1:]:
        record_split=re.split(',|:|;',CC_this_pair.strip())
        pair_name=record_split[0]
        merged_CC_table_read[pair_name]={}
        #record_split should have 2+nstation*len(CC_one_sta) columns (see more detailed description of 
        #merged_CC_table in para_result_analysis.py).
        #Otherwise, something is wrong either in the merged_CC_table or nstation.
        nsta_this_pair=(len(record_split)-2)/(len(CC_one_sta)+1)
        if nstation!=nsta_this_pair:
            sys.exit('Error, given nstation is not consistent with that in merged_CC_table!')

        for ista in range(nstation):
            icolumn_start=1+ista*(len(CC_one_sta)+1)
            station_name=record_split[icolumn_start]
            wave_type=record_split[icolumn_start+1]
            CC_coeff=record_split[icolumn_start+2]
            dif_ref_time=record_split[icolumn_start+3]
            time_delay=record_split[icolumn_start+4]
            CC_error=record_split[icolumn_start+5]
            CC_warning=record_split[icolumn_start+6]
            CC_one_sta={'wave_type':wave_type,'CC':CC_coeff,'dif_ref_time':dif_ref_time,'time_delay':time_delay,'error':CC_error,'warning':CC_warning}
            #print 'pair,sta',pair_name,station_name
            merged_CC_table_read[pair_name][station_name]=CC_one_sta
    return merged_CC_table_read

 
def date_to_julian_day(date):
    """Returns the Julian day number of a date."""
    a = (14 - date.month)//12
    y = date.year + 4800 - a
    m = date.month + 12*a - 3
    return date.day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045


def SNR_amp_peak_to_peak(st,noise_refer_time='t1',t0_noise=-40.0,t1_noise=-10.0, \
                                   signal_refer_time='t1',t0_signal=-10.0,t1_signal=10.0):
################compute the signal-noise ratio (SNR)###################################


#################Signal###############################
    signal_refer_time_to_b = find_time_relative_to_b(st,signal_refer_time)
    signal_t0_utc = st.stats.starttime + signal_refer_time_to_b + t0_signal
    signal_t1_utc = st.stats.starttime + signal_refer_time_to_b + t1_signal
    cut_signal_data = st.slice(signal_t0_utc, signal_t1_utc)
    peaktopeak_amp_signal=max(cut_signal_data) - min(cut_signal_data)



################Noise################################
    noise_refer_time_to_b = find_time_relative_to_b(st,noise_refer_time)
    noise_t0_utc = st.stats.starttime + noise_refer_time_to_b + t0_noise
    noise_t1_utc = st.stats.starttime + noise_refer_time_to_b + t1_noise
    cut_noise_data = st.slice(noise_t0_utc, noise_t1_utc)
    peaktopeak_amp_noise=max(cut_noise_data) - min(cut_noise_data)

################SNR##################################
#adding a tiny value TINY_TOL in the denominator to avoid zero devision.
    signal_noise_ratio=peaktopeak_amp_signal/(peaktopeak_amp_noise+global_var.TINY_TOL)


    return signal_noise_ratio

def find_second_largest(arr):
    # Use numpy.unique to get the unique values in the array (unique_values) and their corresponding counts (counts)
    unique_values, counts = np.unique(arr, return_counts=True)
    
    # Sort the counts in descending order using numpy.argsort
    sorted_indices = np.argsort(-counts)
    
    try:
        # Find the index of the (second) most frequent element in the sorted indices
        second_most_index = sorted_indices[1]
        most_index = sorted_indices[0]

        # Retrieve the (second) most frequent element using the original unique values array
        second_most = unique_values[second_most_index]
        most = unique_values[most_index]
        
        # Return the second most frequent element
        return second_most, most
    except IndexError:
        # if all the numbers are same
        most_index = sorted_indices[0]
        most = unique_values[most_index]
        
        # Return the most frequent element
        return most, -12345
    