import os
import numpy as np
import pandas as pd
from os.path import exists
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import seaborn as sns; sns.set_theme()
import matplotlib.gridspec as gridspec
import subprocess
import time
from google.cloud import storage
from tensorflow.python.lib.io import file_io
import io
import math
import statistics

def getCrewInt(crewID):
	if (crewID == 'Crew_01'):
		b = 1
	elif (crewID == 'Crew_02'):
		b = 2
	elif (crewID == 'Crew_03'):
		b = 3
	elif (crewID == 'Crew_04'):
		b = 4
	elif (crewID == 'Crew_05'):
		b = 5
	elif (crewID == 'Crew_06'):
		b = 6
	elif (crewID == 'Crew_07'):
		b = 7
	elif (crewID == 'Crew_08'):
		b = 8
	elif (crewID == 'Crew_09'):
		b = 9
	elif (crewID == 'Crew_10'):
		b = 10
	elif (crewID == 'Crew_11'):
		b = 11
	elif (crewID == 'Crew_12'):
		b = 12
	elif (crewID == 'Crew_13'):
		b = 13
	return b

crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
crews_to_process = ['Crew_13']
file_types = ["smarteye_leftseat","smarteye_rightseat"]
scenarios = ["1","2","3","5","6","7"]
# scenarios = ["1"]
storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

df = pd.DataFrame()

## CAPT.sew
# InstPanel_capt_lowerleft = [-.54,-532]
# InstPanel_capt_widthheight = [1.57,.37]
kneePanel_capt_lowerleft = [0 * 1000, -.9 * 1000]
kneePanel_capt_widthheight = [.541 * 1000, .1 * 1000]
FIMD_capt_lowerleft = [.03 * 1000, .11 * 1000]
FIMD_capt_widthheight = [.095 * 1000, .075 * 1000]
PFD_capt_lowerleft = [.16 * 1000, .04 * 1000]
PFD_capt_widthheight = [.21 * 1000, .21 * 1000]
Nav_capt_lowerleft = [.385 * 1000, 0.04 * 1000]
Nav_capt_width_height = [.21 * 1000, .21 * 1000]
UpperEICAS_capt_lowerleft = [.69 * 1000, 0.04 * 1000]
UpperEICAS_capt_widthheight = [.21 * 1000, .21 * 1000]
KneePanel_fo_lowerleft = [1.05 * 1000, -0.09 * 1000]
KneePanel_fo_widthheight = [.522 * 1000, 0.1 * 1000]
Nav_fo_lowerleft = [1 * 1000, 0.04 * 1000]
Nav_fo_widthheight = [.21 * 1000, 0.21 * 1000]
PFD_fo_lowerleft = [1.23 * 1000, 0.04 * 1000]
PFD_fo_widthheight = [.21 * 1000, 0.21 * 1000]
FIMD_fo_lowerleft = [1.459 * 1000, .11 * 1000]
FIMD_fo_widthheight = [.095* 1000, .075* 1000]

EFB_capt_lowerleft = [.37* 1000,-.422* 1000]
EFB_capt_widthheight = [.095* 1000, .075* 1000]

CDUpanel_capt_lowerleft = [-.55,-.662]
CDUpanel_capt_widthheight = [.51,.28]

CDUcapt_capt_lowerleft = [0.14, .125]
CDUcapt_capt_widthheight = [0.103, .092]
CDUfo_capt_lowerleft = [.384, .125]
CDUfo_capt_widthheight = [0.103, .092]
MFD_capt_lowerleft = [.384, .125]
MFD_capt_widthheight = [0.103, .092]

leftseat_heatmap = np.zeros((100,100,len(scenarios)))
rightseat_heatmap = np.zeros((100,100,len(scenarios)))

plot_heatmap_and_qatable = 0 # embeds pct_usable value too
vertical_mm = 1575
horizontal_mm = 400
number_of_epochs = 1000
time_per_epoch_4_analysis = 10

for i_crew in range(len(crews_to_process)):
	pct_usable_matrix = np.zeros((len(scenarios),len(file_types)))
	crew_dir = crews_to_process[i_crew]
	process_dir_name = crew_dir + "/Processing/"

	pupild_leftseat = np.zeros((number_of_epochs,len(scenarios)))
	pupild_rightseat = np.zeros((number_of_epochs,len(scenarios)))
	headHeading_leftseat = np.zeros((number_of_epochs,len(scenarios)))
	headHeading_rightseat = np.zeros((number_of_epochs,len(scenarios)))
	smarteye_timesec_epoch_storage = np.zeros((len(scenarios),number_of_epochs))
	# event_smarteyeTime_metrics = np.zeros((4,3,len(file_types),len(scenarios)))

	event_smarteyeTimeSeries_metrics = pd.DataFrame()
	event_smarteyeTime_metrics = np.zeros((len(scenarios)*2,15))
	event_smarteyeTime_metrics[:, 0] = getCrewInt(crews_to_process[i_crew])
	event_smarteyeTime_column_values = ['crew', 'seat', 'scenario', 'headHeading_avg', 'headHeading_std', 'pupilD_avg', 'pupilD_std']

	f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.npy', 'rb')
	this_event_data = np.load(io.BytesIO(f_stream.read()))

	for i_scenario in range(len(scenarios)):
		# if (getCrewInt(crews_to_process[i_crew]) != 13) & 
		if (scenarios[i_scenario] != '5'):
			print("Processing Crew: " + crews_to_process[i_crew] + " Scenario: "+scenarios[i_scenario])
			for i_seat in range(len(file_types)):
				if (i_seat == 0):
					event_smarteyeTime_metrics[i_scenario*2, 1] = 0
					event_smarteyeTime_metrics[i_scenario*2, 2] = scenarios[i_scenario]
				else:
					event_smarteyeTime_metrics[i_scenario*2+1, 1] = 1
					event_smarteyeTime_metrics[i_scenario*2+1, 2] = scenarios[i_scenario]

				blob = bucket.blob(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				if blob.exists():
					smarteye_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
					
					time_vector = np.array(smarteye_data.UserTimeStamp[2:])
					HeadRotationQ = np.array(smarteye_data.HeadRotationQ[2:])
					PupilDiameterQ = np.array(smarteye_data.PupilDiameterQ[2:])

					headheading_good_indices = np.squeeze(np.where(HeadRotationQ > .6))
					pupilD_good_indices = np.squeeze(np.where(PupilDiameterQ > .4))				

					headheading = np.array(smarteye_data.HeadHeading[2:])

					headheadingDeg = headheading * 180/math.pi
					time_diff = np.diff(time_vector)
					
					pupilD = np.array(smarteye_data.PupilDiameter[2:])

					headheadingDeg_rate = np.zeros(headheadingDeg.shape[0])
					headheadingDeg_diff = np.diff(headheadingDeg)
					for this_frame in range(headheadingDeg.shape[0]-1):
						if time_diff[this_frame] != 0.0:
							headheadingDeg_rate[this_frame+1] = headheadingDeg_diff[this_frame] / time_diff[this_frame]
						else:
							headheadingDeg_rate[this_frame+1] = 0

					# HeadHeading_rate_mean - avg 1st 2 min, avg 1st event, avg 2nd event
					# HeadHeading_rate_var - avg 1st 2 min, avg 1st event, avg 2nd event
					# PupilD_mean - avg 1st 2 min, avg 1st event, avg 2nd event
					# PupilD_std - avg 1st 2 min, avg 1st event, avg 2nd event
					if (scenarios[i_scenario] != '7'):
						# need to find the indices 1 minute before and 1 minute after (start and end of event epoch)
						event1_epoch_start = this_event_data[0, i_scenario] - 60
						difference_array = np.absolute(time_vector-event1_epoch_start)
						event1_start_index = difference_array.argmin()
						event1_epoch_end = this_event_data[0, i_scenario] + 60
						difference_array = np.absolute(time_vector-event1_epoch_end)
						event1_end_index = difference_array.argmin()
						event2_epoch_start = this_event_data[1, i_scenario] - 60
						difference_array = np.absolute(time_vector-event2_epoch_start)
						event2_start_index = difference_array.argmin()
						event2_epoch_end = this_event_data[1, i_scenario] + 60
						difference_array = np.absolute(time_vector-event2_epoch_end)
						event2_end_index = difference_array.argmin()
						difference_array = np.absolute(time_vector-120)
						two_min_index = difference_array.argmin()

						headheading_event1_good_indices = np.squeeze(np.where((headheading_good_indices > event1_start_index) & (headheading_good_indices < event1_end_index)))
						headheading_event2_good_indices = np.squeeze(np.where((headheading_good_indices > event2_start_index) & (headheading_good_indices < event2_end_index)))
						headheading_twomin_good_indices = np.squeeze(np.where((headheading_good_indices < two_min_index)))

						pupilD_event1_good_indices = np.squeeze(np.where((pupilD_good_indices > event1_start_index) & (pupilD_good_indices < event1_end_index)))
						pupilD_event2_good_indices = np.squeeze(np.where((pupilD_good_indices > event2_start_index) & (pupilD_good_indices < event2_end_index)))
						pupilD_twomin_good_indices = np.squeeze(np.where((pupilD_good_indices < two_min_index)))

						# event_smarteyeTime_metrics[0,0,i_seat,i_scenario] = statistics.mean(headheadingDeg_rate[headheading_twomin_good_indices])
						# event_smarteyeTime_metrics[1,0,i_seat,i_scenario] = statistics.stdev(headheadingDeg_rate[headheading_twomin_good_indices])
						# event_smarteyeTime_metrics[2,0,i_seat,i_scenario] = statistics.mean(pupilD[pupilD_twomin_good_indices])
						# event_smarteyeTime_metrics[3,0,i_seat,i_scenario] = statistics.stdev(pupilD[pupilD_twomin_good_indices])

						# event_smarteyeTime_metrics[0,1,i_seat,i_scenario] = statistics.mean(headheadingDeg_rate[headheading_event1_good_indices])
						# event_smarteyeTime_metrics[1,1,i_seat,i_scenario] = statistics.stdev(headheadingDeg_rate[headheading_event1_good_indices])
						# event_smarteyeTime_metrics[2,1,i_seat,i_scenario] = statistics.mean(pupilD[pupilD_event1_good_indices])
						# event_smarteyeTime_metrics[3,1,i_seat,i_scenario] = statistics.stdev(pupilD[pupilD_event1_good_indices])

						# event_smarteyeTime_metrics[0,2,i_seat,i_scenario] = statistics.mean(headheadingDeg_rate[headheading_event2_good_indices])
						# event_smarteyeTime_metrics[1,2,i_seat,i_scenario] = statistics.stdev(headheadingDeg_rate[headheading_event2_good_indices])
						# event_smarteyeTime_metrics[2,2,i_seat,i_scenario] = statistics.mean(pupilD[pupilD_event2_good_indices])
						# event_smarteyeTime_metrics[3,2,i_seat,i_scenario] = statistics.stdev(pupilD[pupilD_event2_good_indices])

						if (i_seat == 0):
							event_smarteyeTime_metrics[i_scenario*2, 3] = statistics.mean(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 4] = statistics.mean(headheadingDeg_rate[headheading_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 5] = statistics.mean(headheadingDeg_rate[headheading_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 6] = statistics.stdev(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 7] = statistics.stdev(headheadingDeg_rate[headheading_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 8] = statistics.stdev(headheadingDeg_rate[headheading_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 9] = statistics.mean(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 10] = statistics.mean(pupilD[pupilD_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 11] = statistics.mean(pupilD[pupilD_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 12] = statistics.stdev(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 13] = statistics.stdev(pupilD[pupilD_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 14] = statistics.stdev(pupilD[pupilD_event2_good_indices])
						else:
							event_smarteyeTime_metrics[i_scenario*2+1, 3] = statistics.mean(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 4] = statistics.mean(headheadingDeg_rate[headheading_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 5] = statistics.mean(headheadingDeg_rate[headheading_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 6] = statistics.stdev(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 7] = statistics.stdev(headheadingDeg_rate[headheading_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 8] = statistics.stdev(headheadingDeg_rate[headheading_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 9] = statistics.mean(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 10] = statistics.mean(pupilD[pupilD_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 11] = statistics.mean(pupilD[pupilD_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 12] = statistics.stdev(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 13] = statistics.stdev(pupilD[pupilD_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 14] = statistics.stdev(pupilD[pupilD_event2_good_indices])

						number_of_epochs_this_scenario = np.floor(time_vector.shape[0]/time_per_epoch_4_analysis)
						this_smarteyeTimeSeries_np = np.zeros((int(number_of_epochs_this_scenario), 9))
						# event_smarteyeTime_column_values = ['crew', 'seat', 'scenario', 'event_label', 'headHeading_avg', 'headHeading_std', 'pupilD_avg', 'pupilD_std']
						this_smarteyeTimeSeries_np[:,0] = getCrewInt(crews_to_process[i_crew])
						if (i_seat == 0):
							this_smarteyeTimeSeries_np[:,1] = 0
						else:
							this_smarteyeTimeSeries_np[:,1] = 1
						this_smarteyeTimeSeries_np[:,2] = i_scenario
						for this_epoch in range(int(number_of_epochs_this_scenario)):
							if ((time_vector[10*this_epoch] > this_event_data[0, i_scenario] - 60) & (time_vector[10*this_epoch] < this_event_data[0, i_scenario] + 60)) | ((time_vector[10*this_epoch] > this_event_data[1, i_scenario] - 60) & (time_vector[10*this_epoch] < this_event_data[1, i_scenario] + 60)):
								this_smarteyeTimeSeries_np[this_epoch, 3] = 1
							else:
								this_smarteyeTimeSeries_np[this_epoch, 3] = 0

							this_smarteyeTimeSeries_np[this_epoch, 4] = this_epoch
							this_smarteyeTimeSeries_np[this_epoch, 5] = np.nanmean(headheadingDeg_rate[10*this_epoch:10*this_epoch + 10])
							this_smarteyeTimeSeries_np[this_epoch, 6] = np.nanstd(headheadingDeg_rate[10*this_epoch:10*this_epoch + 10])
							this_smarteyeTimeSeries_np[this_epoch, 7] = np.nanmean(pupilD[10*this_epoch:10*this_epoch + 10])
							this_smarteyeTimeSeries_np[this_epoch, 8] = np.nanstd(pupilD[10*this_epoch:10*this_epoch + 10])
						this_smarteyeTimeSeries_df = pd.DataFrame(this_smarteyeTimeSeries_np)
						this_smarteyeTimeSeries_df.columns = ['crew', 'seat', 'scenario', 'event_label', 'epoch_index','headHeading_avg', 'headHeading_std', 'pupilD_avg', 'pupilD_std']
						event_smarteyeTimeSeries_metrics = event_smarteyeTimeSeries_metrics.append(this_smarteyeTimeSeries_df)
					else:
						difference_array = np.absolute(time_vector - ((time_vector[-1]/2) - 150))
						fivemin_start_index = difference_array.argmin()
						difference_array = np.absolute(time_vector - ((time_vector[-1]/2) + 150))
						fivemin_stop_index = difference_array.argmin()

						headheading_fivmin_good_indices = np.squeeze(np.where((headheading_good_indices > fivemin_start_index) & (headheading_good_indices < fivemin_stop_index)))
						pupilD_fivmin_good_indices = np.squeeze(np.where((pupilD_good_indices > fivemin_start_index) & (pupilD_good_indices < fivemin_stop_index)))

						if (i_seat == 0):
							event_smarteyeTime_metrics[i_scenario*2, 3] = statistics.mean(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 6] = statistics.stdev(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 9] = statistics.mean(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 12] = statistics.stdev(pupilD[pupilD_twomin_good_indices])
						else:
							event_smarteyeTime_metrics[i_scenario*2+1, 3] = statistics.mean(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 6] = statistics.stdev(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 9] = statistics.mean(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 12] = statistics.stdev(pupilD[pupilD_twomin_good_indices])
												

						# event_smarteyeTime_metrics[0,0,i_seat,i_scenario] = statistics.mean(headheadingDeg_rate[headheading_fivmin_good_indices])
						# event_smarteyeTime_metrics[1,0,i_seat,i_scenario] = statistics.stdev(headheadingDeg_rate[headheading_fivmin_good_indices])
						# event_smarteyeTime_metrics[2,0,i_seat,i_scenario] = statistics.mean(pupilD[pupilD_fivmin_good_indices])
						# event_smarteyeTime_metrics[3,0,i_seat,i_scenario] = statistics.stdev(pupilD[pupilD_fivmin_good_indices])


					length_this_data = smarteye_data.shape[0]
					for this_epoch in range(number_of_epochs):
						this_epoch_indices_start = np.floor(length_this_data/number_of_epochs) * this_epoch
						this_epoch_indices_end = this_epoch_indices_start + np.floor(length_this_data/number_of_epochs)
						smarteye_timesec_epoch_storage[i_scenario,this_epoch] = smarteye_data.UserTimeStamp[this_epoch_indices_start]
						if file_types[i_seat]  == "smarteye_leftseat":
							pupild_leftseat[this_epoch,i_scenario] = smarteye_data.PupilDiameter[int(this_epoch_indices_start):int(this_epoch_indices_end)].mean()
							headHeading_leftseat[this_epoch,i_scenario] = smarteye_data.HeadHeading[int(this_epoch_indices_start):int(this_epoch_indices_end)].mean()
							
						elif file_types[i_seat]  == "smarteye_rightseat":
							pupild_rightseat[this_epoch,i_scenario] = smarteye_data.PupilDiameter[int(this_epoch_indices_start):int(this_epoch_indices_end)].mean()
							headHeading_rightseat[this_epoch,i_scenario] = smarteye_data.HeadHeading[int(this_epoch_indices_start):int(this_epoch_indices_end)].mean()
					
					if plot_heatmap_and_qatable:
						x = smarteye_data.ObjectIntersectionX * 1000 # m to mm?
						y = smarteye_data.ObjectIntersectionY * 1000 # m to mm? where is the origin?
						length_this_trial = smarteye_data.shape[0]
						data_quality_vector = np.ones(length_this_trial)
						for i_index in range(smarteye_data.shape[0]):
							if smarteye_data.IntersectionIndex[i_index] == 0 or smarteye_data.GazeDirectionQ[i_index] < .50:
								x[i_index] = 0
								y[i_index] = 0
								data_quality_vector[i_index] = 0
						pct_usable_matrix[i_scenario,i_seat] = np.rint((np.sum(data_quality_vector)/length_this_trial) * 100)
						x[ x==0 ] = np.nan
						y[ y==0 ] = np.nan
						x = x[~np.isnan(x)]
						y = y[~np.isnan(y)]
					else:
						# print("Empty: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
						pct_usable_matrix[i_scenario,i_seat] = np.nan
						
						if plot_heatmap_and_qatable:
							empty_heatmap = np.zeros((100,100))
							if file_types[i_seat] == "smarteye_leftseat":
								leftseat_heatmap[:,:,i_scenario] = empty_heatmap
							elif file_types[i_seat] == "smarteye_rightseat":
								rightseat_heatmap[:,:,i_scenario] = empty_heatmap

	# if plot_heatmap_and_qatable:
	# 	plt.subplots_adjust(wspace=0, hspace=0)
	# 	plt.style.use("dark_background")
	# 	# plt.show()
	# 	plt.savefig("Figures/" + 'smarteye_'+crews_to_process[i_crew]+'.tif',bbox_inches='tight',pad_inches=0)
	# 	# Instead of plotting individual tables (embed percent text in subplot or in title of subplot?)
	# 	matplotlib.pyplot.close()

	if exists("Figures"):
		subprocess.Popen('rm -rf Figures', shell=True)
		time.sleep(5)
		os.mkdir("Figures")
	else:
		os.mkdir("Figures")
	if exists("Processing"):
		subprocess.Popen('rm -rf Processing', shell=True)
		time.sleep(5)
		os.mkdir("Processing")
	else:
		os.mkdir("Processing")	

	np.save("Processing/" + 'smarteye_pupild_leftseat',pupild_leftseat)
	np.save("Processing/" + 'smarteye_pupild_rightseat',pupild_rightseat)
	np.save("Processing/" + 'smarteye_headHeading_leftseat',headHeading_leftseat)
	np.save("Processing/" + 'smarteye_headHeading_rightseat',headHeading_rightseat)
	np.save("Processing/" + 'smarteye_timesec_epoch_storage',smarteye_timesec_epoch_storage)
	np.save("Processing/" + 'event_smarteyeTime_metrics', event_smarteyeTime_metrics)
	event_smarteyeTimeSeries_metrics.info()
	event_smarteyeTimeSeries_metrics.to_csv("Processing/" + 'event_smarteyeTimeSeries_metrics.csv')
	subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Processing"', shell=True)

	print('should have saved')


	# fig, ax = plt.subplots()
	# cbar_kws = { 'ticks' : [0, 100] }
	# ax = sns.heatmap(pct_usable_matrix, linewidths=.5, cbar_kws = cbar_kws,annot=True,fmt='.3g')
	# ax.set_xticklabels(file_types)
	# ax.set_yticklabels(scenarios)
	# ax.set(xlabel='pilot', ylabel='scenarios')
	# plt.yticks(rotation=0) 
	# ax.xaxis.set_label_position('top') 
	# ax.xaxis.tick_top()
	# fig.tight_layout()
	# # plt.show()
	# plt.savefig(crew_dir + "/Figures/" + 'smarteye_pct_usable.jpg')
	# matplotlib.pyplot.close()
	# np.save(crew_dir + "/Processing/" + 'pct_usable_matrix',pct_usable_matrix)

	# np.save(crew_dir + "/Processing/" + 'smarteye_leftseat_heatmap',leftseat_heatmap)
	# np.save(crew_dir + "/Processing/" + 'smarteye_rightseat_heatmap',rightseat_heatmap)
	# scenarioAvg_leftseat_heatmap = leftseat_heatmap.mean(2)
	# scenarioAvg_rightseat_heatmap = rightseat_heatmap.mean(2) 

	# plt.imshow(scenarioAvg_leftseat_heatmap, extent=extent, origin='lower')
	# plt.axis('off')
	# # plt.show()
	# plt.savefig(crew_dir + "/Figures/" + 'smarteye_leftseat_scenarioAvg.tif',bbox_inches='tight',pad_inches=0)
	# # os.remove(crew_dir + "/Figures/" + 'smarteye_leftseat_scenarioAvg.jpg')
	# matplotlib.pyplot.close()
	# plt.imshow(scenarioAvg_rightseat_heatmap, extent=extent, origin='lower')
	# plt.axis('off')
	# # plt.show()
	# plt.savefig(crew_dir + "/Figures/" + 'smarteye_rightseat_scenarioAvg.tif',bbox_inches='tight',pad_inches=0)
	# # os.remove(crew_dir + "/Figures/" + 'smarteye_rightseat_scenarioAvg.jpg')
	# matplotlib.pyplot.close()