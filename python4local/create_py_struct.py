
import pandas as pd
import os
import array as arr
import awkward as ak
import numpy as np
from os.path import exists
from datetime import datetime
from google.cloud import storage
#bucket_name = os.getenv("soteria_study_data")

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
#bucket = storage_client.bucket("soteria_study_data")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")
#all_blobs = list(storage_client.list_blobs(bucket))
#print(all_blobs)

#crews_to_process = ['Crew1', 'Crew2', 'Crew3', 'Crew4', 'Crew5', 'Crew6', 'Crew7', 'Crew8', 'Crew9', 'Crew10', 'Crew11', 'Crew12', 'Crew13']
crews_to_process = ['Crew1']
#print(crews_to_process)
#path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'
###############################################

for this_crew in crews_to_process:
	#all_blobs = list(storage_client.list_blobs(bucket))
	#all_blobs.shape
	#print(all_blobs)
	#blob = bucket.blob("Crew1") # this would be the file name
	#blob.shape
	crew_dir = this_crew
	#crew_dir.shape
	
	#print(crew_dir)
	#trial_folders = os.listdir(crew_dir + '/Synched')
	#print(trial_folders)
	blob = bucket.blob(crew_dir + '/trial_settings.txt')
	blob.download_to_filename("test2")
	
	trial_settings = pd.read_table(crew_dir + '/trial_settings.txt',delimiter=',')
	print(trial_settings)
	for this_folder in range(trial_settings.shape[0]):
		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/ABM.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/ABM.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/ABM.log'),delimiter='\t')
			if not this_table.empty:
				abm_leftseat = adjust_timestamps(this_table)
			
		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Acc.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Acc.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Acc.log'),delimiter='\t')
			if not this_table.empty:
				emp_acc_leftseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Bvp.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Bvp.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Bvp.log'),delimiter='\t')
			if not this_table.empty:
				emp_bvp_leftseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Gsr.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Gsr.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Gsr.log'),delimiter='\t')
			if not this_table.empty:
				emp_gsr_leftseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Ibi.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Ibi.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Ibi.log'),delimiter='\t')
			if not this_table.empty:
				emp_ibi_leftseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Temp.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device_Temp.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Temp.log'),delimiter='\t')
			if not this_table.empty:
				emp_temp_leftseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Left_Seat_ET.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/SmartEye_Left_Seat_ET.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Left_Seat_ET.log'),delimiter='\t')
			if not this_table.empty:
				smarteye_leftseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/ABM-1.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/ABM-1.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/ABM-1.log'),delimiter='\t')
			if not this_table.empty:
				abm_rightseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Acc.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Acc.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Acc.log'),delimiter='\t')
			if not this_table.empty:
				emp_acc_rightseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Bvp.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Bvp.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Bvp.log'),delimiter='\t')
			if not this_table.empty:
				emp_bvp_rightseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Gsr.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Gsr.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Gsr.log'),delimiter='\t')
			if not this_table.empty:
				emp_gsr_rightseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Ibi.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Ibi.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Ibi.log'),delimiter='\t')
			if not this_table.empty:
				emp_ibi_rightseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Temp.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/Emp_Emp_Device2_Temp.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Temp.log'),delimiter='\t')
			if not this_table.empty:
				emp_temp_rightseat = adjust_timestamps(this_table)	

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Right_Seat_ET.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/SmartEye_Right_Seat_ET.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Right_Seat_ET.log'),delimiter='\t')
			if not this_table.empty:
				smarteye_rightseat = adjust_timestamps(this_table)

		if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/IFD_COCKPIT.log'):
			print("processing: " + (crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder])+ '/IFD_COCKPIT.log'))
			this_table = pd.read_table((crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/IFD_COCKPIT.log'),delimiter='\t')
			if not this_table.empty:
				ifd_cockpit = adjust_timestamps(this_table)


		if not os.path.isdir(crew_dir + "/Analysis"):
			os.mkdir(crew_dir + "/Analysis")

		crew_dir_split = crew_dir.split('/')
		this_crew = crew_dir_split[-1]

		with open(crew_dir + "/Analysis" + this_crew + '_'+ str(trial_settings.Scenario[this_folder]) +'.npy', 'wb') as f:
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/ABM.log'):
				np.save(f, exec("%s_%s = abm_leftseat" % ('abm_leftseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Acc.log'):
				np.save(f, exec("%s_%s = emp_acc_leftseat" % ('emp_acc_leftseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Bvp.log'):
				np.save(f, exec("%s_%s = emp_bvp_leftseat" % ('emp_bvp_leftseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Gsr.log'):
				np.save(f, exec("%s_%s = emp_gsr_leftseat" % ('emp_gsr_leftseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Ibi.log'):
				np.save(f, exec("%s_%s = emp_ibi_leftseat" % ('emp_ibi_leftseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device_Temp.log'):
				np.save(f, exec("%s_%s = emp_temp_leftseat" % ('emp_temp_leftseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Left_Seat_ET.log'):   
				np.save(f, exec("%s_%s = smarteye_leftseat" % ('smarteye_leftseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/ABM-1.log'):
				np.save(f, exec("%s_%s = abm_rightseat" % ('abm_rightseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Acc.log'):
				np.save(f, exec("%s_%s = emp_acc_rightseat" % ('emp_acc_rightseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Bvp.log'):
				np.save(f, exec("%s_%s = emp_bvp_rightseat" % ('emp_bvp_rightseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Gsr.log'):    
				np.save(f, exec("%s_%s = emp_gsr_rightseat" % ('emp_gsr_rightseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Ibi.log'):
				np.save(f, exec("%s_%s = emp_ibi_rightseat" % ('emp_ibi_rightseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/Emp_Emp_Device2_Temp.log'):
				np.save(f, exec("%s_%s = emp_temp_rightseat" % ('emp_temp_rightseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/SmartEye_Right_Seat_ET.log'):
				np.save(f, exec("%s_%s = smarteye_rightseat" % ('smarteye_rightseat',str(trial_settings.Scenario[this_folder]))))
			if exists(crew_dir + "/Synched/" + str(trial_settings.RunDateTime[this_folder]) + '/IFD_COCKPIT.log'):
				np.save(f, exec("%s_%s = ifd_cockpit" % ('ifd_cockpit',str(trial_settings.Scenario[this_folder]))))

		

		

		
		

		

		
	#os.chdir(crew_dir)



	# with open('test.npy', 'rb') as f:
	#     a = np.load(f)
	#     b = np.load(f)
	#for this_folder in trial_folders:
	#print(this_folder)
		#pd.read_table('SmartEye_Left_Seat_ET.log', delimiter='\t')

