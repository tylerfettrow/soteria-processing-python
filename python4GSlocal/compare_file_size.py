import os
import pandas as pd
from os.path import exists
import shutil
################################# check file size locall ################################

#crews_to_process = ['Crew1', 'Crew2', 'Crew3', 'Crew4', 'Crew5', 'Crew6', 'Crew7', 'Crew8', 'Crew9', 'Crew10', 'Crew11', 'Crew12', 'Crew13']
crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04', 'Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
file_types = ["ifd_cockpit", "smarteye_leftseat","smarteye_rightseat", "abm_leftseat","abm_rightseat","emp_acc_leftseat","emp_acc_rightseat","emp_bvp_leftseat","emp_bvp_rightseat","emp_gsr_leftseat","emp_gsr_rightseat","emp_ibi_leftseat","emp_ibi_rightseat","emp_temp_leftseat","emp_temp_rightseat"];
scenarios = ["1","2","3","5","6","7","8","9"];

path_to_project_2 = 'C:/Users/tfettrow/Box/SOTERIA/'
path_to_project_1 = 'D:/SOTERIA/SOTERIA_Study_Data/'

for this_crew in crews_to_process:

	trial_settings = pd.read_table(path_to_project_2 + "/" + this_crew + '/trial_settings.txt',delimiter=',')
	
	for this_folder in range(trial_settings.shape[0]):
		dir_name_1 = path_to_project_1 + this_crew + "/Synched/" + trial_settings.RunDateTime[this_folder]+ "/"
		dir_name_2 = path_to_project_2 + this_crew + "/Synched/" + trial_settings.RunDateTime[this_folder]+ "/"
		
		#if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + 

		file_list_1 = os.listdir(dir_name_1)
		#print(file_list_1)
		#file_list_2 = os.listdir(dir_name_2
		
		for file_name in file_list_1:
			full_file_path_1 = dir_name_1 + file_name
			full_file_path_2 = dir_name_2 + file_name

			if exists(full_file_path_2):
				file_stats_1 = os.stat(full_file_path_1)
				file_stats_2 = os.stat(full_file_path_2)
				if file_stats_1.st_size == file_stats_2.st_size:
					print(1)
				else:
					# print(0)
					print("copying: " + full_file_path_1)
					shutil.copy2(full_file_path_1, full_file_path_2)
			else:
				# print(0)
				print("copying: " + full_file_path_1)
				shutil.copy2(full_file_path_1, full_file_path_2)
			# print(file_stats_1.st_size)
			# print(file_stats_2.st_size)

		

