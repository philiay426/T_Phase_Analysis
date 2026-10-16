import sys
import numpy as np

TINY_TOL=1.e-16
HUGE_TOL=1.e20
#HUGE_TOL=sys.float_info.max

#The 1D Earth model used to predict the arrival times
Earthmodel_1D="prem"


#great circle distance for a spherical or elliptical Earth
gcarc_spherical=True
R_Eearth=6371000.0

#We keep t0 as a temporary variable, which might be used in the codes. 
#So do not put any information you want to save permanently into t0.
tmp_time='t0'
#the variable names of arrival times of various phases, predicted by 1D Earth model
#We wrap them in a dictionary of similar objects, that makes querying their values much easier.
t_P_S_1DEarth = {
    'P':'t1',
    'S':'t2',
    }

#The factor used in waveform interpolation. dt_original -> dt_origional/interpo_factor
interpo_factor=2

#after processing the below number of traces, print the progress.
print_prog_every_ntraces=2

#parameters for CC
max_shift_time_CC=8.0
#cut the time window 'between time_begin_CC' seconds before the P-wave to 
#length_CC_unit_PtoS_time*PtoS_time seconds after the P-wave to do CC.
time_begin_CC=2.0
length_CC_unit_PtoS_time=6.0
#If the cut-window ls longer than 50 seconds, use 50 seconds.
length_CC_max=30.0
#length_CC_max=80.0

#The thresholds to determine whether or not do the CC.
#If the paried events have too large location difference in ISC catalog, 
#we know they cannot be repeating pair.
dif_hypo_threshold_km=60.0
dif_dep_threshold_km=60.0

#The CC threshold, above which we consider it as a good pair
high_CC_threshold=0.5

#the folder path to save CC waveform figures
OUTPUT_PATH='OUTPUT/'
FILE_SAVE_ALL_PAIRS='all_CC_results.txt'
FILE_SAVE_GOOD_PAIRS='good_evts_CC_results.txt'







#####################For measuring T-wave time delays####################
#the folder path to save CC result and figures
UTPUT_PATH_Twaves='OUTPUT/'

#the variable names of arrival times of various phases.
#e.g. t3 for the arrival time of T-waves
arrival_Twaves_trace= {
          'Twaves':'t3',
    }


##########
#The time window length used in CC is defined by 'window_length_CC_Twaves'. 
#The first CC computation cuts the waveforms, which start at the arrival time of Twaves (e.g. 
#t3 defined in the dictionary 'arrival_Twaves_trace') and ends at t3+window_length_CC_Twaves.
#We slide this time window by a amount of 'time_sliding_Twaves' seconds for the next CC. 
#And this sliding is repeated 'nwindows_Twaves' times for all CC computations.
#Thus, the whole length of time window involved in CCs is 
#(nwindows_Twaves-1)*time_sliding_Twaves+window_length_CC_Twaves.

#The length of time window to do CC, in unit of second.
window_length_CC_Twaves=60.0
#the number of windows to do CC.
nwindows_Twaves=1
#the amoung of time to slide the window for next CC computation, in unit of second.
time_sliding_Twaves=1.0
#the allowed maximum time shift in CC.
max_shift_time_CC_Twaves=2.0

#CC using the whole long time window.
#window_length_CC_Twaves=50.0
#nwindows_Twaves=1
#max_shift_time_CC_Twaves=2.0


#####################Searching repeaters using T-waves####################
#because we use the same algorithm to search the event pairs with high CCs, the parameters 
#below are similar to that used in searching repeaters based on local stations' records, 


#after processing the below number of traces, print the progress.
print_prog_every_ntraces_Twaves=2

#parameters for CC
max_time_shift_repeating_Twaves=1.0
#max_time_shift_repeating_Twaves=9.0
#cut the time window between time_begin_CC seconds before the T-wave to 
#length_CC_Twave-time_begin_CC seconds after the T-wave to do CC.
#Note that a positive number of time_begin_CC_repeating_Twaves means the seconds 
#before the pricked time (e.g. time_begin_CC_repeating_Twaves=20.0 means 20 seconds before the
#the pricked time).
#time_begin_CC_repeating_Twaves=20.0
time_begin_CC_repeating_Twaves=-10.0
#time_begin_CC_repeating_Twaves=0.0
#length_CC_repeating_Twaves=100.0
length_CC_repeating_Twaves=60.0
#length_CC_repeating_Twaves=50.0

#The thresholds to determine whether or not do the CC.
#If the paried events have too large location difference in ISC catalog, 
#we know they cannot be repeating pair.
dif_hypo_threshold_km_Twaves=60.0
dif_dep_threshold_km_Twaves=60.0

#The CC threshold, above which we consider it as a good pair
high_CC_threshold_Twaves=0.6
#high_CC_threshold_Twaves=0.3

#the folder path to save CC waveform figures
OUTPUT_PATH_TWAVES='OUTPUT/'
OUTPUT_PATH_CYCLE_SKIPPING='OUTPUT_cycle_skipping/'
FILE_SAVE_ALL_PAIRS_TWAVES='all_CC_results.txt'
FILE_SAVE_GOOD_PAIRS_TWAVES='good_evts_CC_results.txt'






########################paths of Twaves######################################
#The T-waves are converted from seismic waves at a number of locations at shallow coast, and 
#therefore have multple travel paths.Here, we try to specify each convertion point (e.g. based on
# the distance to the earthquake and bathymetry contaurs) and compute the coordinates (lat,lon) of each path. 

paths_Twaves= {
          'lat_start':0.0,
          'lon_start':0.0,
          'lat_end':0.0,
          'lon_end':0.0,
          'path':[],
          'weight':0.0,
          'dist_starting_to_eq':0.0,
    }

#the distance increment of path array
ddist_path=50.0

#deps_Twave_sampled=[25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 245.0, 250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0, 425.0, 450.0]
#deps_Twave_sampled=np.linspace(16.0,1950.0,100)
deps_Twave_sampled=np.linspace(16.0,5000.0,250)
#deps_Twave_sampled=[25.0, 50.0, 75.0]

Twave_station= {"DGAR":{"lat":-7.4121,"lon":72.4525},"PALK":{"lat":7.2728,"lon":80.7022}, "ATOC_k":{"lat":42.08,"lon":171.12}, "WAKE":{"lat":19.283,"lon":166.652},"MAT":{"lat":-14.87,"lon":-148.712006}}


Argo_starting_year=2004
Argo_dep_min=0.0
Argo_dep_max=2000.0




########################SNR of T waves#####################################
#define the noise time window for SNR_Twave, between -t0_noise and -t1_noise before the Twave arrival time.
t0_Twave_noise=-200.0
t1_Twave_noise=-50.0

#define the Twave time window, between -t0_S2KS_SNR before and t1_S2KS_SNR after the S2KS arrival. Usually, t0_S2KS_SNR is negative to ensure the S2KS included. 
t0_Twave_SNR=-50.0
t1_Twave_SNR=100.0

##############################hydrophone trinet array analysis################
consistency_threshold=0.06
# correlation_threshold=0.0
dspeed_search=5.0
speed_min=1340.0
speed_max=1570.0
# speed_min=1480.0
# speed_max=1490.0
dazimuth_search=0.2

