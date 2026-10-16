#ls *EDH*bp3_5 > file_list.txt

#To run the below script, 
#(1) make sure you have installed the module 'obspy'.
#(2) mark the reference time in the sac headers. Here, the scipt uses 't1' as the reference times. 
#python azimuth_speed_array.py  file_list starting_time time_length_one_slice sliding_time total_time_length
python azimuth_speed_array.py  ./file_list/file_list_09_wsl.txt 2000.0 20.0 5.0 5000.0
