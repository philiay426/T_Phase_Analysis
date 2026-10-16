import sys
import numpy as np

#The Twave and seismic stations used.
#e.g. station_list=["DGAR","KUM","WRAB"]
station_list=[]

#merged_CC_table has the CC results from all the events and stations. Its structure is like merged_CC_table ={"key_pair_name":{"sta1":CC_one_sta,"sta2":CC_one_sta}}
#e.g. CC_table_all_evt_sta = {"20050329214949.69-20050621041931.76":{"DGAR":{'wave_type':'T','CC':0.68,'dif_ref_time':0.0,'time_delay':-0.51,'error_tag':0,'warning_tag':4}
#                                                                    "KUM":{'wave_type':'NULL','CC':0.0,'dif_ref_time':0.0,'time_delay':0.0,'error_tag':0,'warning_tag':0} 
#                                                                    "WRAB":{'wave_type':'PS','CC':0.72,'dif_ref_time':-0.65,'time_delay':-0.596,'error_tag':0,'warning_tag':0}
#                                                                   }
#                             "20051225024915.31-20100916033915.17":{"DGAR":{'wave_type':'T','CC':0.71,'dif_ref_time':0.0,'time_delay':-0.0907,'error_tag':0,'warning_tag':0}
#                                                                    "KUM":{'wave_type':'PS','CC':0.83,'dif_ref_time':2.57,'time_delay':0.1235,'error_tag':0,'warning_tag':0} 
#                                                                    "WRAB":{'wave_type':'NULL','CC':0.0,'dif_ref_time':0.0,'time_delay':0.0,'error_tag':0,'warning_tag':0} 
#                                                                   }
#                            }
merged_CC_table = {}


CC_one_sta={'wave_type':'NULL',
        'CC':0.0,
        'dif_ref_time':0.0,
        'time_delay':0.0,
        'error_tag':0,
        'warning_tag':0,}
