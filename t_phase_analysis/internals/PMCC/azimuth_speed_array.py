import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from math import cos
#TOT - T-waves and Ocean Temperature
import global_parameters_PMCC as global_var
from PMCC_auxiliary import  read_trace_list, find_time_relative_to_b, find_second_largest
#from obspy.geodetics.base import  gps2dist_azimuth
import os
import os.path
from obspy import read
import obspy
from obspy.geodetics.base import  gps2dist_azimuth
from cross_correlation_WENBO import xcorr_pick_correction



def main():
    """
    #################Function#############
    #Compute the CC and time-shift betwee event pair

    ################INPUT##################
    Three parameters - 
    1) type file_traces_workon: string
       param file_traces_workon: The names of work-on sac traces to do PMCC.
    2) type starting_time: float, in unit of second.
       param starting_time: The starting time to do PMCC. The reference time is 'b' in the sac header.
    3) type time_one_slice: float, in unit of second.
       param time_one_slice: The time length of one slice (one mearsure point in the final plotting) to do PMCC.     
    4) type sliding_time: float, in unit of second.
       param sliding_time: We do it in a sliding time window way, so this parameter gives the shifting  amount of time to get the next slice.
    4) type time_length: float, in unit of second.
       param time_length: The total time length of PMCC.

    #########OUTPUT#########################
    Output: CC results of each sliding time windows

    Usage:
    python $path_this_file/azimuth_speed_array.py file_list.txt -10.0 10.0 2.0 100.0 
    """

    file_trace_list=sys.argv[1]
#    file_trace_workon=sys.argv[2]
    time_start=float(sys.argv[2])
    time_one_slice=float(sys.argv[3])
    time_slide=float(sys.argv[4])
    time_length=float(sys.argv[5])
#    delay_trace_workon=float(sys.argv[3])


    #read the two T-wave traces you want to work on.
    trace_list_workon = read_trace_list(file_trace_list)
    if(len(trace_list_workon)!=3):
        sys.exit('Only three traces allowed!')
    nsta=3
    for itrace in range(len(trace_list_workon)):
        if(itrace==0):
           st=read(trace_list_workon[itrace])
        else:
           st+=read(trace_list_workon[itrace])

    file_array_analysis=open('array_analysis_results.txt','w+')
  
 
    print('working on each time window...')


    #parameters used in CC. 
    #The below parameters are defined in the file global_parameters_TOT.py. 
    #See more details there.
    max_shift_time_CC=2.0
    #the variable in sac head (e.g. 't3') to save the T-wave arrival time. 
    var_name_Twave_arrival=global_var.reference_time["Pwave"]
    nwindows=int(time_length/time_slide)

    dt1_2=np.zeros(nwindows);CC1_2=np.zeros(nwindows)
    dt1_3=np.zeros(nwindows);CC1_3=np.zeros(nwindows)
    dt2_3=np.zeros(nwindows);CC2_3=np.zeros(nwindows)
    time_measure=np.zeros(nwindows)

    for ista in range(nsta):
        for ista_workon in range(ista+1,3):
            #Loop for sliding the time window and computing waveform CC.
            for iwindow in range(nwindows):
                #The sliding time of this time window, relative to the arrival time of T-wave.
                time_slide_this_window=iwindow*time_slide
                #The starting time of CC.
                t_pick=st[0].stats.starttime + \
                  find_time_relative_to_b(st[0],var_name_Twave_arrival) + time_start +\
                  time_slide_this_window
                time_measure[iwindow]=find_time_relative_to_b(st[0],var_name_Twave_arrival) + time_start +\
                  time_slide_this_window

                CC_dt=0.0
                CC_coeff=0.0
                error_tag=0
                warning_tag=0

                #CC computation
                CC_dt, CC_coeff, warning_tag= xcorr_pick_correction(t_pick,st[ista], \
                        t_pick,st[ista_workon], \
                        0.0,time_one_slice,max_shift_time_CC, \
                        plot = False,filename='tmp_CC')
     
                if ista==0 and ista_workon==1:
                    dt1_2[iwindow]=-CC_dt
                    CC1_2[iwindow]=CC_coeff
                elif ista==0 and ista_workon==2:
                    dt1_3[iwindow]=-CC_dt
                    CC1_3[iwindow]=CC_coeff
                else:
                    dt2_3[iwindow]=-CC_dt
                    CC2_3[iwindow]=CC_coeff
                #print 'CC',iwindow,CC_dt, CC_coeff
                # starting and ending times of CC, relative to the origin time of the earthquake
                #CC_time_start=sta[ista].stats.sac[var_name_Twave_arrival.lower()]+time_sliding
                #CC_time_end=CC_time_start+time_one_slice



                #save the CC_dt and CC_coeff
                #file_CC.write("template: %s ,target: %s , window: %d th , \
#CC_dt: %8.4f , CC: %6.2f , t_start: %6.2f , t_end: %6.2f , targ_origin_delay: %8.4f, error: %d , warning: %d\n" \
#                % (template_evt_name, workon_evt_name,iwindow,CC_dt,CC_coeff, \
#                   CC_time_start, CC_time_start,delay_trace_workon,error_tag,warning_tag))

    #the azimuth and distance between stations 1, 2 and 3.
    [dist_meter_12, az_12, baz_12] = \
           gps2dist_azimuth(st[0].stats.sac.stla,st[0].stats.sac.stlo,st[1].stats.sac.stla,st[1].stats.sac.stlo)
    [dist_meter_13, az_13, baz_13] = \
           gps2dist_azimuth(st[0].stats.sac.stla,st[0].stats.sac.stlo,st[2].stats.sac.stla,st[2].stats.sac.stlo)
    [dist_meter_23, az_23, baz_23] = \
           gps2dist_azimuth(st[1].stats.sac.stla,st[1].stats.sac.stlo,st[2].stats.sac.stla,st[2].stats.sac.stlo)
    deg2rad=math.pi/180.0

    consistency=np.zeros(nwindows)  
    ave_CC=np.zeros(nwindows)
    mistfit=np.zeros(nwindows)  
    azimuth_find=np.zeros(nwindows)
    speed_find=np.zeros(nwindows)
    for iwindow in range(nwindows):
        consistency[iwindow]=math.fabs(dt1_2[iwindow]+dt2_3[iwindow]-dt1_3[iwindow])
        ave_CC[iwindow]=(CC1_2[iwindow]+CC1_3[iwindow]+CC2_3[iwindow])/3.0
        #search the speed and azimuth
        #We know the ocean sound speed must be in the rage of 1.4-1.55 km/s.
        if consistency[iwindow]<global_var.consistency_threshold: #and ave_CC[iwindow]>global_var.correlation_threshold:
            mistfit[iwindow]=global_var.HUGE_TOL
            for speed in np.arange(global_var.speed_min,global_var.speed_max,global_var.dspeed_search):
                for azimuth_wave in np.arange (0,360.0,global_var.dazimuth_search):
                    dt1_2_predict=-dist_meter_12*cos((az_12-azimuth_wave)*deg2rad)/speed
                    dt1_3_predict=-dist_meter_13*cos((az_13-azimuth_wave)*deg2rad)/speed
                    dt2_3_predict=-dist_meter_23*cos((az_23-azimuth_wave)*deg2rad)/speed
                    mistfit_new=(dt1_2_predict-dt1_2[iwindow])**2+\
                        (dt1_3_predict-dt1_3[iwindow])**2+\
                        (dt2_3_predict-dt2_3[iwindow])**2
#                    print 'consistency1',iwindow,consistency[iwindow],mistfit_new,azimuth_wave,speed,dist_meter_12,dt1_2_predict,dt1_3_predict,dt2_3_predict,dt1_2[iwindow],dt1_3[iwindow],dt2_3[iwindow]
                    if mistfit_new<mistfit[iwindow]:
                        mistfit[iwindow]=mistfit_new
                        if azimuth_wave>180.0:
                            azimuth_find[iwindow]=azimuth_wave-180.0
                        else:
                            azimuth_find[iwindow]=azimuth_wave+180.0
                        speed_find[iwindow]=speed
        else:
            azimuth_find[iwindow]=-12345
            speed_find[iwindow]=-12345

      #calculate the az, baz and dist from station 1 to station 2.
        print('consistency',iwindow,consistency[iwindow],mistfit[iwindow],azimuth_find[iwindow],speed_find[iwindow])

    [azimuth_mode_result_1, azimuth_mode_result_2] = find_second_largest(azimuth_find)
    print(azimuth_mode_result_1)
    print(azimuth_mode_result_2)

    [speed_mode_result_1, speed_mode_result_2] = find_second_largest(speed_find)
    print(speed_mode_result_1)
    print(speed_mode_result_2)

    #draw the results
    fig,axs=plt.subplots(5, figsize=(14, 16))
    plt.subplots_adjust(hspace=0.05)
    
    t0 = st[0].stats.sac['t0']
    
    time_measure = time_measure - t0

    axs[0].scatter(time_measure,consistency)
    axs[0].axhline(global_var.consistency_threshold, color="g", label="threshold")
    axs[0].set_ylabel('consistency (s)', fontsize=25)
    axs[0].set_xticks([])
    axs[0].set_xlim([time_measure[0], time_measure[nwindows-1]])
    axs[0].set_ylim([0.0, 0.06])
    axs[0].legend(loc="upper right", prop={'size': "x-large"})
    axs[0].tick_params(axis='y', labelsize=20)
    
    axs[1].scatter(time_measure,ave_CC)
    # axs[1].axhline(global_var.correlation_threshold, color="g", label="threshold")
    axs[1].set_ylabel('correlation', fontsize=25)
    axs[1].set_xticks([])
    axs[1].set_xlim([time_measure[0], time_measure[nwindows-1]])
    axs[1].set_ylim([0, 1.0])
    axs[1].legend(loc="upper right", prop={'size': "x-large"})
    axs[1].tick_params(axis='y', labelsize=20)


    axs[2].scatter(time_measure,azimuth_find)
    axs[2].set_ylabel('Azimuth (deg)', fontsize=25)
    axs[2].set_xticks([])
    axs[2].set_xlim([time_measure[0], time_measure[nwindows-1]])
    axs[2].set_ylim([0.0, 360.0])
    axs[2].tick_params(axis='y', labelsize=20)
    axs[2].axhline(azimuth_mode_result_1, color='red', linestyle='dashed', label="predicted")
#    axs[2].axhline(74.6, color='black', linestyle='dashed', label="Southern path")
#    axs[2].axhline(66.6, color='green', linestyle='dashed', label="Northern path")


    axs[3].scatter(time_measure,speed_find)
    axs[3].set_ylabel('Speed (m/s)', fontsize=25)
    axs[3].set_xticks([])
    axs[3].set_xlim([time_measure[0], time_measure[nwindows-1]])
    axs[3].set_ylim([global_var.speed_min, global_var.speed_max])
    # axs[3].axhline(speed_mode_result_1, color='red', linestyle='dashed', label="predicted")
    axs[3].tick_params(axis='y', labelsize=20)


    tstart_plot=st[0].stats.starttime + time_measure[0] + t0
    tend_plot=st[0].stats.starttime + time_measure[nwindows-1] + t0
    st[0].trim(tstart_plot,tend_plot)
    waveform=st[0].data
    time_series=np.arange(0,len(waveform),1)*st[0].stats.delta+time_measure[0]
    axs[4].plot(time_series,waveform,color='black')
#    axs[4].axvline(time_measure[0]+10.0, color="red",linestyle='dashed')
#    axs[4].axvline(time_measure[0]+70.0, color="red",linestyle='dashed')

    axs[4].set_xlim([time_measure[0], time_measure[nwindows-1]])
    station_name=st[0].stats.station
    axs[4].set_ylabel(station_name, fontsize=25)
    axs[4].set_xlabel('Time after origin (s)', fontsize=25)
    print('time',time_measure[0],time_measure[nwindows-1])
    axs[4].tick_params(axis='x', labelsize=20)
    axs[4].tick_params(axis='y', labelsize=20)

    # 保存到当前目录
    base = f'n_{var_name_Twave_arrival}_{int(time_start):d}_{int(time_one_slice):d}_{int(time_slide):d}_{int(time_length):d}'
    plt.savefig(f'{base}.png', dpi=200, bbox_inches='tight')

    np.savez('n_Time_measure_'+var_name_Twave_arrival+'_'+str(int(time_start))+'_'+str(int(time_one_slice))+'_'+str(int(time_slide))+'_'+str(int(time_length))+'.npz', data = time_measure)
    np.savez('n_Consistency_'+var_name_Twave_arrival+'_'+str(int(time_start))+'_'+str(int(time_one_slice))+'_'+str(int(time_slide))+'_'+str(int(time_length))+'.npz', data = consistency)
    np.savez('n_ave_CC_'+var_name_Twave_arrival+'_'+str(int(time_start))+'_'+str(int(time_one_slice))+'_'+str(int(time_slide))+'_'+str(int(time_length))+'.npz', data = ave_CC)
    np.savez('n_Azimuth_find_'+var_name_Twave_arrival+'_'+str(int(time_start))+'_'+str(int(time_one_slice))+'_'+str(int(time_slide))+'_'+str(int(time_length))+'.npz', data = azimuth_find)
    np.savez('n_Speed_find_'+var_name_Twave_arrival+'_'+str(int(time_start))+'_'+str(int(time_one_slice))+'_'+str(int(time_slide))+'_'+str(int(time_length))+'.npz', data = speed_find)

    
#close the output files
#    file_CC.close()

 
if __name__ == "__main__":
    main()

