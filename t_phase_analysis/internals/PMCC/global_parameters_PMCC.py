import sys
import numpy as np

TINY_TOL=1.e-16
HUGE_TOL=1.e20
#HUGE_TOL=sys.float_info.max

#the variable names of arrival times of various phases.
#e.g. t3 for the arrival time of T-waves
reference_time= {
          'Pwave':'t0',
    }

##############################hydrophone trinet array analysis################
consistency_threshold=0.05
correlation_threshold=0.3
dspeed_search=1.0
# speed_min=1480.0
# speed_max=1490.0
speed_min=1200.0
speed_max=1800.0

dazimuth_search=0.1
