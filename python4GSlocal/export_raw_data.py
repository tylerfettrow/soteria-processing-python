import pandas as pd
import os
import array as arr
import numpy as np
from os.path import exists
from datetime import datetime
#from google.cloud import storage
from google.cloud import storage
import subprocess
import time
from pathlib import Path
#bucket_name = os.getenv("soteria_study_data")
import shutil

#print("APPDAT SIMPLE STORAGE SERVICE | Currently connected to Bucket: " + bucket_name)


def adjust_timestamps(datainput):
	timestamps_time = np.zeros(len(datainput.UserTimeStamp))
	for this_index in range(datainput.UserTimeStamp.shape[0]):
		this_timestamp = datainput.UserTimeStamp[this_index]/1e7
		this_timestamp.astype('int64')
		this_timestamp_datetime = datetime.fromtimestamp(this_timestamp)
		this_timestamp_time_string = str(this_timestamp_datetime.time())

		## H:M:S -> seconds
		this_timestamp_time_string_split = this_timestamp_time_string.split(':')
		timestamps_time[this_index] = float(this_timestamp_time_string_split[0]) * 3600 + float(this_timestamp_time_string_split[1]) * 60 + float(this_timestamp_time_string_split[2])

	timestamps_time_adjusted = timestamps_time - timestamps_time[0]	
	datainput.UserTimeStamp = timestamps_time_adjusted
	return datainput


###############################################
## need to adjust this for gsutil ##
#crew_dir = os.getcwd()

storage_client = storage.Client(project="soteria-fa59")
bucket = storage_client.bucket("soteria_study_data")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")
#all_blobs = list(storage_client.list_blobs(bucket))
#print(all_blobs)

crews_to_process = ['Crew_04', 'Crew_05', 'Crew_09']
# crews_to_process = ['Crew_13']
# crews_to_process = ['Crew_03']
#print(crews_to_process)
#path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'
###############################################

for this_crew in crews_to_process:
	# crew_dir = path_to_project + '/' + this_crew
	crew_dir = this_crew
	
	# blob = bucket.blob(crew_dir + '/trial_settings.txt')
	col_Names = ["RunDateTime","Scenario"]
	trial_settings = pd.read_table('gs://soteria_study_data/' + crew_dir + '/trial_settings.txt',delimiter=",")

	# Get blobs in bucket (including all subdirectories)
	# blobs_all = list(bucket.list_blobs())

	# Get blobs in specific subirectory
	# blobs_specific = bucket.list_blobs(prefix=crew_dir + '/Synched/', delimiter="/",max_results=1)
	
	# print("prefixes:")
	# print(list(blobs_specific.prefixes))
	# trial_folders = list(blobs_specific.prefixes)
	
	
	# if not os.path.isdir('gs://soteria_study_data/'+crew_dir + "/Processing"):
		# print("Processing")
		# 'gs://soteria_study_data/Crew_02/trial_settings.txt'
		# os.mkdir('gs://soteria_study_data/'+crew_dir + "/Processing")


	# gsutil -q stat 'gs://soteria_study_data/'+crew_dir + "/Processing"; echo $?
	# dir_name = 'gs://soteria_study_data/'+crew_dir + "/Processing"
	# file_list = os.listdir(dir_name)
	
	# subprocess.Popen('gsutil rm "gs://soteria_study_data/"' + crew_dir + '"/Processing/ghost.txt"', shell=True)
	# blob = bucket.blob(crew_dir + '/Processing')
	# if blob.exists():
	# 	print("yep")
	# else:

	if exists("Processing"):
		subprocess.Popen('rm -rf Processing', shell=True)
		time.sleep(5)
		os.mkdir("Processing")
	else:
		os.mkdir("Processing")

	for this_folder in range(trial_settings.shape[0]):
		blob = bucket.blob(crew_dir + '/Synched/' + str(trial_settings.RunDateTime[this_folder]) + '/ABM.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/ABM.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + '/Synched/' + str(trial_settings.RunDateTime[this_folder]) + '/ABM.log'),delimiter='\t')
			if not this_table.empty:
				abm_leftseat = adjust_timestamps(this_table)
				abm_leftseat.to_csv("Processing/" + 'abm_leftseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Acc.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Acc.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Acc.log'),delimiter='\t')
			if not this_table.empty:
				emp_acc_leftseat = adjust_timestamps(this_table)
				emp_acc_leftseat.to_csv("Processing/" + 'emp_acc_leftseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Bvp.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Bvp.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Bvp.log'),delimiter='\t')
			if not this_table.empty:
				emp_bvp_leftseat = adjust_timestamps(this_table)
				emp_bvp_leftseat.to_csv("Processing/" + 'emp_bvp_leftseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Gsr.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Gsr.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' +crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Gsr.log'),delimiter='\t')
			if not this_table.empty:
				emp_gsr_leftseat = adjust_timestamps(this_table)
				emp_gsr_leftseat.to_csv("Processing/" + 'emp_gsr_leftseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Ibi.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Ibi.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' +crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Ibi.log'),delimiter='\t')
			if not this_table.empty:
				emp_ibi_leftseat = adjust_timestamps(this_table)
				emp_ibi_leftseat.to_csv("Processing/" + 'emp_ibi_leftseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Temp.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Temp.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Temp.log'),delimiter='\t')
			if not this_table.empty:
				emp_temp_leftseat = adjust_timestamps(this_table)
				emp_temp_leftseat.to_csv("Processing/" + 'emp_temp_leftseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Left_Seat_ET.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/SmartEye_Left_Seat_ET.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Left_Seat_ET.log'),delimiter='\t')
			if not this_table.empty:
				smarteye_leftseat = adjust_timestamps(this_table)
				smarteye_leftseat.to_csv("Processing/" + 'smarteye_leftseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/ABM-1.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/ABM-1.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/ABM-1.log'),delimiter='\t')
			if not this_table.empty:
				abm_rightseat = adjust_timestamps(this_table)
				abm_rightseat.to_csv("Processing/" + 'abm_rightseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Acc.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Acc.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Acc.log'),delimiter='\t')
			if not this_table.empty:
				emp_acc_rightseat = adjust_timestamps(this_table)
				emp_acc_rightseat.to_csv("Processing/" + 'emp_acc_rightseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Bvp.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Bvp.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Bvp.log'),delimiter='\t')
			if not this_table.empty:
				emp_bvp_rightseat = adjust_timestamps(this_table)
				emp_bvp_rightseat.to_csv("Processing/" + 'emp_bvp_rightseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Gsr.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Gsr.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Gsr.log'),delimiter='\t')
			if not this_table.empty:
				emp_gsr_rightseat = adjust_timestamps(this_table)
				emp_gsr_rightseat.to_csv("Processing/" + 'emp_gsr_rightseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Ibi.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Ibi.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Ibi.log'),delimiter='\t')
			if not this_table.empty:
				emp_ibi_rightseat = adjust_timestamps(this_table)
				emp_ibi_rightseat.to_csv("Processing/" + 'emp_ibi_rightseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Temp.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Temp.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Temp.log'),delimiter='\t')
			if not this_table.empty:
				emp_temp_rightseat = adjust_timestamps(this_table)
				emp_temp_rightseat.to_csv("Processing/" + 'emp_temp_rightseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Right_Seat_ET.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/SmartEye_Right_Seat_ET.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Right_Seat_ET.log'),delimiter='\t')
			if not this_table.empty:
				smarteye_rightseat = adjust_timestamps(this_table)
				smarteye_rightseat.to_csv("Processing/" + 'smarteye_rightseat_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

		blob = bucket.blob(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/IFD_COCKPIT.log')
		if blob.exists():
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/IFD_COCKPIT.log'))
			this_table = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/IFD_COCKPIT.log'),delimiter='\t')
			if not this_table.empty:
				ifd_cockpit = adjust_timestamps(this_table)
				ifd_cockpit.to_csv("Processing/" + 'ifd_cockpit_' + str(trial_settings.Scenario[this_folder])+'.csv',index=False)

	# subprocess.Popen('gsutil cp -r Processing/ "gs://soteria_study_data/"'+ crew_dir + '"/"', shell=True)
	subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crew_dir + '"/Processing"', shell=True)
	# time.sleep(60)