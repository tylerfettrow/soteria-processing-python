import os
import shutil
from os.path import exists
# import glob

crews_to_process = ['Crew_02', 'Crew_03', 'Crew_04', 'Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = ['Crew_01']
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'

# processing_filenames = ["smarteye_pupild_leftseat","smarteye_pupild_rightseat","smarteye_headHeading_leftseat","smarteye_headHeading_rightseat","smarteye_timesec_band_storage","eeg_freq_band_storage_leftseat","eeg_timesec_band_storage_leftseat","eeg_freq_band_storage_rightseat","eeg_timesec_band_storage_rightseat","eeg_freq_storage_leftseat","eeg_freq_storage_rightseat","ecg_bpm_leftseat","ecg_timesec_band_storage","ecg_bpm_rightseat","event_vector_scenario","abm_leftseat_","emp_acc_leftseat_","emp_bvp_leftseat_","emp_gsr_leftseat_","emp_ibi_leftseat_","emp_temp_leftseat_","smarteye_leftseat_"]

for i_crew in range(len(crews_to_process)):
	if exists(path_to_project+'/'+crews_to_process[i_crew]+'/Processing'):
		shutil.rmtree(path_to_project+'/'+crews_to_process[i_crew]+'/Processing')
	if exists(path_to_project+'/'+crews_to_process[i_crew]+'/Figures'):
		shutil.rmtree(path_to_project+'/'+crews_to_process[i_crew]+'/Figures')
	if exists(path_to_project+'/'+crews_to_process[i_crew]+'/Analysis'):
		shutil.rmtree(path_to_project+'/'+crews_to_process[i_crew]+'/Analysis')
			
