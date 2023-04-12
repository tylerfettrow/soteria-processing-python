import pandas as pd
import os
import array as arr
import numpy as np
import numpy.matlib
from google.cloud import storage
import subprocess
import time
import math
import matplotlib.pyplot as plt
from numpy import linalg as la
from os.path import exists
from sklearn.neighbors._kde import KernelDensity
from distinctipy import distinctipy
from tensorflow.python.lib.io import file_io
import io

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

def sphere_stereograph(p):
	# p = np.squeeze(direction_gaze[:,good_indices])

	[m,n]=p.shape

	s = np.divide(2.0, ( 1.0 + p[m-1,0:n] ))
	ss = np.matlib.repmat( s, m, 1 )

	f = np.zeros((m,1))
	f[m-1] = -1.0
	ff = np.matlib.repmat( f, 1, n )

	q = np.multiply(ss, p) + np.multiply(( 1.0 - ss ), ff);

	b = q[0:2,:]

	return b


def angle_diff(time_vector,direction_gaze):
	degree_per_sec_vector = np.zeros(direction_gaze.shape[1])

	time_diff = np.diff(time_vector)
	for this_frame in range(direction_gaze.shape[1]-1):
		if time_diff[this_frame] != 0.0:
			degree_diff_vector = math.degrees(2 * math.atan(la.norm(np.multiply(direction_gaze[:,this_frame],la.norm(direction_gaze[:,this_frame+1])) - np.multiply(la.norm(direction_gaze[:,this_frame]),direction_gaze[:,this_frame+1])) / la.norm(np.multiply(direction_gaze[:,this_frame], la.norm(direction_gaze[:,this_frame+1])) + np.multiply(la.norm(direction_gaze[:,this_frame]), direction_gaze[:,this_frame+1]))))
			degree_per_sec_vector[this_frame+1] = degree_diff_vector / time_diff[this_frame]
		else:
			degree_per_sec_vector[this_frame+1] = 0
	return degree_per_sec_vector

def mean(a, n):
 
    # Calculating sum
    sum = 0;
    for i in range(n):
        for j in range(n):
            sum += a[i][j];
     
    # Returning mean
    return sum / (n * n);
 
# Function for calculating variance
def variance(a, n, m):
    sum = 0;
    for i in range(n):
        for j in range(n):
 
            # subtracting mean
            # from elements
            a[i][j] -= m;
 
            # a[i][j] = fabs(a[i][j]);
            # squaring each terms
            a[i][j] *= a[i][j];
 
    # taking sum
    for i in range(n):
        for j in range(n):
            sum += a[i][j];
 
    return sum / (n * n);

crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
crews_to_process = ['Crew_13']
file_types = ["smarteye_leftseat", "smarteye_rightseat"]
scenarios = ["1","2","3","5","6","7"]
time_per_epoch_4_analysis = 10

storage_client = storage.Client(project="soteria-fa59")
bucket = storage_client.bucket("soteria_study_data")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

plt.style.use('dark_background')
for i_crew in range(len(crews_to_process)):
	process_dir_name = crews_to_process[i_crew] + "/Processing/"
	
	event_smarteyeGazeTimeSeries_metrics = pd.DataFrame()
	total_gaze_variance_matrix = np.zeros((len(file_types),len(scenarios)))
	total_gaze_velocity_avg_matrix = np.zeros((len(file_types),len(scenarios)))
	total_gaze_velocity_std_matrix = np.zeros((len(file_types),len(scenarios)))
	# event_smarteyeGaze_metrics = np.zeros((3,3,len(file_types),len(scenarios)))

	event_smarteyeGaze_metrics = np.zeros((len(scenarios)*2,12))
	event_smarteyeGaze_metrics[:, 0] = getCrewInt(crews_to_process[i_crew])
	event_smarteyeGaze_column_values = ['crew', 'seat', 'scenario', 'gaze_variance_control', 'gaze_variance_event1', 'gaze_variance_event2', 'gaze_vel_avg_control', 'gaze_vel_avg_event1', 'gaze_vel_avg_event2', 'gaze_vel_std_control', 'gaze_vel_std_event1', 'gaze_vel_std_event2']

	f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.npy', 'rb')
	this_event_data = np.load(io.BytesIO(f_stream.read()))
	# this_event_data = pd.read_table(('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.csv'),delimiter=',')
	# this_event_data = this_event_data.to_numpy()

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
		
	for i_scenario in range(len(scenarios)):
		# if (getCrewInt(crews_to_process[i_crew]) != 13) & 
		if (scenarios[i_scenario] != '5'):
			print("Processing Crew: " + crews_to_process[i_crew] + " Scenario: " +scenarios[i_scenario])
			for i_seat in range(len(file_types)):
				if (i_seat == 0):
					event_smarteyeGaze_metrics[i_scenario*2, 1] = 0
					event_smarteyeGaze_metrics[i_scenario*2, 2] = scenarios[i_scenario]
				else:
					event_smarteyeGaze_metrics[i_scenario*2+1, 1] = 1
					event_smarteyeGaze_metrics[i_scenario*2+1, 2] = scenarios[i_scenario]
				
				blob = bucket.blob(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				if blob.exists():
					print("QA checking: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
					smarteye_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
					object_data = smarteye_data.ObjectIntersectionName.values

					time_vector = np.array(smarteye_data.UserTimeStamp[2:])
					
					direction_gaze = np.array([smarteye_data.GazeDirectionX[2:], smarteye_data.GazeDirectionY[2:], smarteye_data.GazeDirectionZ[2:]])
					magnitude = np.divide(np.sqrt(np.power(direction_gaze[0,:],2) + np.power(direction_gaze[1,:],2) + np.power(direction_gaze[2,:],2)),1)
					quality_gaze = smarteye_data.GazeDirectionQ[2:]
					degree_per_sec_vector = angle_diff(time_vector, direction_gaze)

					# for the indices that are sequential (i.e. no gaps in good_indices), calculate rate of gaze movement, and remove values that are greater than 700°/s ( Fuchs, A. F. (1967-08-01). "Saccadic and smooth pursuit eye movements in the monkey". The Journal of Physiology. 191 (3): 609–631. doi:10.1113/jphysiol.1967.sp008271. ISSN 1469-7793. PMC 1365495. PMID 4963872.)
					# unit circle 1 -> -1
					# WARNING: not quite what I intended ^^
					good_indices = np.squeeze(np.where((magnitude!=0) & (quality_gaze*100 >= 6) & (degree_per_sec_vector<=700)))

					# projected_planar_coords = sphere_stereograph(np.squeeze(direction_gaze[:,good_indices]))
					projected_planar_coords = sphere_stereograph(np.squeeze(direction_gaze))
					good_project_planar_coords = projected_planar_coords[:,good_indices]

					object_data_good = np.squeeze(object_data[good_indices])
					unique_objects = np.unique(object_data_good.tolist())

					total_good_vel_vals = np.squeeze(degree_per_sec_vector[good_indices]) 
					# exclude the first since that was set to 0 on purpose
					total_average_gaze_velocity = np.average(total_good_vel_vals[1:])
					total_std_gaze_velocity = np.std(total_good_vel_vals[1:])

					# https://www.geeksforgeeks.org/variance-standard-deviation-matrix/
					total_m = mean(projected_planar_coords[:,good_indices],2)
					total_gaze_variance = variance(projected_planar_coords[:,good_indices], 2, total_m)

					# for the event_smarteyeSphere.npy ... and if scenario 7, take middle 5 minutes and store in first column
					# gaze_variance - avg 1st 2 min, avg 1st event, avg 2nd event
					# gaze_vel_avg - avg 1st 2 min, avg 1st event, avg 2nd event
					# gaze_vel_std - avg 1st 2 min, avg 1st event, avg 2nd event
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

						event1_good_indices = np.squeeze(np.where((good_indices > event1_start_index) & (good_indices < event1_end_index)))
						event2_good_indices = np.squeeze(np.where((good_indices > event2_start_index) & (good_indices < event2_end_index)))
						twomin_good_indices = np.squeeze(np.where((good_indices < two_min_index)))

						twomin_mean = mean(projected_planar_coords[:,twomin_good_indices],2)
						# event_smarteyeGaze_metrics[0,0,i_seat,i_scenario] = variance(projected_planar_coords[:,twomin_good_indices], 2, twomin_mean)

						event1_mean = mean(projected_planar_coords[:,event1_good_indices],2)
						# event_smarteyeGaze_metrics[0,1,i_seat,i_scenario] = variance(projected_planar_coords[:,event1_good_indices], 2, event1_mean)
						
						event2_mean = mean(projected_planar_coords[:,event2_good_indices],2)
						# event_smarteyeGaze_metrics[0,2,i_seat,i_scenario] = variance(projected_planar_coords[:,event2_good_indices], 2, event2_mean)
						
						twomin_good_vel_vals = np.squeeze(degree_per_sec_vector[twomin_good_indices])
						event1_good_vel_vals = np.squeeze(degree_per_sec_vector[event1_good_indices])
						event2_good_vel_vals = np.squeeze(degree_per_sec_vector[event2_good_indices])

						# event_smarteyeGaze_metrics[1,0,i_seat,i_scenario] = np.average(twomin_good_vel_vals[1:])
						# event_smarteyeGaze_metrics[2,0,i_seat,i_scenario] = np.std(twomin_good_vel_vals[1:])
						# event_smarteyeGaze_metrics[1,1,i_seat,i_scenario] = np.average(event1_good_vel_vals[1:])
						# event_smarteyeGaze_metrics[2,1,i_seat,i_scenario] = np.std(event1_good_vel_vals[1:])
						# event_smarteyeGaze_metrics[1,2,i_seat,i_scenario] = np.average(event2_good_vel_vals[1:])
						# event_smarteyeGaze_metrics[2,2,i_seat,i_scenario] = np.std(event2_good_vel_vals[1:])

						if (i_seat == 0):
							event_smarteyeGaze_metrics[i_scenario*2, 3] = variance(projected_planar_coords[:,twomin_good_indices], 2, twomin_mean)
							event_smarteyeGaze_metrics[i_scenario*2, 4] = variance(projected_planar_coords[:,event1_good_indices], 2, event1_mean)
							event_smarteyeGaze_metrics[i_scenario*2, 5] = variance(projected_planar_coords[:,event2_good_indices], 2, event2_mean)
							event_smarteyeGaze_metrics[i_scenario*2, 6] = np.nanmean(twomin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 7] = np.nanmean(event1_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 8] = np.nanmean(event2_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 9] = np.nanstd(twomin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 10] = np.nanstd(event1_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 11] = np.nanstd(event2_good_vel_vals[1:])
						else:
							event_smarteyeGaze_metrics[i_scenario*2+1, 3] = variance(projected_planar_coords[:,twomin_good_indices], 2, twomin_mean)
							event_smarteyeGaze_metrics[i_scenario*2+1, 4] = variance(projected_planar_coords[:,event1_good_indices], 2, event1_mean)
							event_smarteyeGaze_metrics[i_scenario*2+1, 5] = variance(projected_planar_coords[:,event2_good_indices], 2, event2_mean)
							event_smarteyeGaze_metrics[i_scenario*2+1, 6] = np.nanmean(twomin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 7] = np.nanmean(event1_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 8] = np.nanmean(event2_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 9] = np.nanstd(twomin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 10] = np.nanstd(event1_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 11] = np.nanstd(event2_good_vel_vals[1:])

						number_of_epochs_this_scenario = np.floor(time_vector.shape[0]/time_per_epoch_4_analysis)
						this_smarteyeGazeTimeSeries_np = np.zeros((int(number_of_epochs_this_scenario), 8))
						# event_smarteyeTime_column_values = ['crew', 'seat', 'scenario', 'event_label', 'headHeading_avg', 'headHeading_std', 'pupilD_avg', 'pupilD_std']
						this_smarteyeGazeTimeSeries_np[:,0] = getCrewInt(crews_to_process[i_crew])
						if (i_seat == 0):
							this_smarteyeGazeTimeSeries_np[:,1] = 0
						else:
							this_smarteyeGazeTimeSeries_np[:,1] = 1
						this_smarteyeGazeTimeSeries_np[:,2] = i_scenario
						for this_epoch in range(int(number_of_epochs_this_scenario)):
							if ((time_vector[10*this_epoch] > this_event_data[0, i_scenario] - 60) & (time_vector[10*this_epoch] < this_event_data[0, i_scenario] + 60)) | ((time_vector[10*this_epoch] > this_event_data[1, i_scenario] - 60) & (time_vector[10*this_epoch] < this_event_data[1, i_scenario] + 60)):
								this_smarteyeGazeTimeSeries_np[this_epoch, 3] = 1
							else:
								this_smarteyeGazeTimeSeries_np[this_epoch, 3] = 0
							this_smarteyeGazeTimeSeries_np[this_epoch, 4] = this_epoch
							this_good_indices = np.squeeze(np.where((good_indices > 10*this_epoch) & (good_indices < 10*this_epoch + 10)))
							if (this_good_indices.size > 1):
								this_smarteyeGazeTimeSeries_np[this_epoch, 5] = variance(projected_planar_coords[:,this_good_indices], 2, mean(projected_planar_coords[:,this_good_indices],1))
								this_smarteyeGazeTimeSeries_np[this_epoch, 6] = np.nanmean(np.squeeze(degree_per_sec_vector[this_good_indices]))
								this_smarteyeGazeTimeSeries_np[this_epoch, 7] = np.nanstd(np.squeeze(degree_per_sec_vector[this_good_indices]))
							else:
								this_smarteyeGazeTimeSeries_np[this_epoch, 5] = np.nan
								this_smarteyeGazeTimeSeries_np[this_epoch, 6] = np.nan
								this_smarteyeGazeTimeSeries_np[this_epoch, 7] = np.nan
						this_smarteyeGazeTimeSeries_df = pd.DataFrame(this_smarteyeGazeTimeSeries_np)
						this_smarteyeGazeTimeSeries_df.columns = ['crew', 'seat', 'scenario', 'event_label', 'epoch_index', 'gaze_variance', 'gaze_vel_avg', 'gaze_vel_std']

						event_smarteyeGazeTimeSeries_metrics = event_smarteyeGazeTimeSeries_metrics.append(this_smarteyeGazeTimeSeries_df)
					else:
						difference_array = np.absolute(time_vector - ((time_vector[-1]/2) - 150))
						fivemin_start_index = difference_array.argmin()
						difference_array = np.absolute(time_vector - ((time_vector[-1]/2) + 150))
						fivemin_stop_index = difference_array.argmin()

						fivemin_good_indices = np.squeeze(np.where((good_indices > fivemin_start_index) & (good_indices < fivemin_stop_index)))

						fivemin_mean = mean(projected_planar_coords[:,fivemin_good_indices],2)
						
						fivemin_good_vel_vals = np.squeeze(degree_per_sec_vector[fivemin_good_indices])

						if (i_seat == 0):
							event_smarteyeGaze_metrics[i_scenario*2, 3] = variance(projected_planar_coords[:,twomin_good_indices], 2, fivemin_mean)
							event_smarteyeGaze_metrics[i_scenario*2, 6] = np.nanmean(fivemin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 9] = np.nanstd(fivemin_good_vel_vals[1:])
						else:
							event_smarteyeGaze_metrics[i_scenario*2+1, 3] = variance(projected_planar_coords[:,twomin_good_indices], 2, fivemin_mean)
							event_smarteyeGaze_metrics[i_scenario*2+1, 6] = np.nanmean(fivemin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 9] = np.nanstd(fivemin_good_vel_vals[1:])



					objects_of_interest = np.array(['Inst_Panel_Capt', 'Inst_Panel_FO',
			       'EFB', 'CDU_Panel_Capt', 'CDU_Panel_FO', 'MCP_Capt',
			       'MCP_FO', 'OutTheWindow','OverheadPanel'])

					colors = [(0.0, 0.5, 1.0),
					(1.0, 0.5, 0.0),
					(0.6392495878543843, 0.022349116680812298, 0.5050150900480403),
					(0.0628137825307874, 0.8806313927830106, 0.5332618165496811),
					(0.9994394377998805, 0.2608117991316806, 0.9611791323454995),
					(0.0, 0.5, 0.0),
					(0.0, 0.0, 1.0),
					(0.0, 1.0, 1.0),
					(0.984865806821456, 0.8385889426164463, 0.05610936943298106)]
					# distinctipy.color_swatch(colors)


					if (i_seat == 0):
						fig2 = plt.figure(2)

						unique_objects_used_indices = np.zeros([0])
						for i_object in range(len(objects_of_interest)):
							if (objects_of_interest[i_object] == 'Inst_Panel_Capt'):
								indices_this_object_of_interest = np.where((object_data_good == 'Inst Panel') | (object_data_good == 'Inst Panel.KNEE_PANEL_CAPT') | (object_data_good == 'Inst Panel.FIMD_Capt')
								 | (object_data_good == 'Inst Panel.PFD_Capt') | (object_data_good == 'Inst Panel.PFD_Capt_Airspd') | (object_data_good == 'Inst Panel.PFD_Capt_ADI') | (object_data_good == 'Inst Panel.PFD_Capt_Altitude')
								 | (object_data_good == 'Inst Panel.PFD_Capt_FMA') | (object_data_good == 'Inst Panel.Nav_Capt') | (object_data_good == 'Inst Panel.UPPER_EICAS'))
							elif (objects_of_interest[i_object] == 'Inst_Panel_FO'):
								indices_this_object_of_interest = np.where((object_data_good == 'Inst Panel.KNEE_PANEL_FO') | (object_data_good == 'Inst Panel.Nav_FO') | (object_data_good == 'Inst Panel.PFD_FO') | (object_data_good == 'Inst Panel.FIMD_FO'))
							elif (objects_of_interest[i_object] == 'EFB_Capt'):
								indices_this_object_of_interest = np.where((object_data_good == 'EFB_Pilot'))
							elif (objects_of_interest[i_object] == 'CDU_Panel_Capt'):
								indices_this_object_of_interest = np.where((object_data_good == 'CDU_Panel') | (object_data_good == 'CDU_Panel_Capt.CDU_screen_Capt') | (object_data_good == 'CDU_Panel_Capt.CDU_Capt') | (object_data_good == 'CDU_Panel_Capt.FMC_Capt')| (object_data_good == 'CDU_Panel_Capt.MFD'))
							elif (objects_of_interest[i_object] == 'CDU_Panel_FO'):
								indices_this_object_of_interest = np.where((object_data_good == 'CDU_Panel_Capt.CDU_screen_FO') | (object_data_good == 'CDU_Panel_Capt.CDU_FO') )
							elif (objects_of_interest[i_object] == 'MCP_Capt'):
								indices_this_object_of_interest = np.where((object_data_good == 'MCP') | (object_data_good == 'MCP.EDCP_Capt'))
							elif (objects_of_interest[i_object] == 'MCP_FO'):
								indices_this_object_of_interest = np.where((object_data_good == 'MCP.EDCP_FO'))	
							elif (objects_of_interest[i_object] == 'OutTheWindow'):
								indices_this_object_of_interest = np.where((object_data_good == 'OutTheWindow'))
							elif (objects_of_interest[i_object] == 'OverheadPanel'):
								indices_this_object_of_interest = np.where((object_data_good == 'OverheadPanel'))	

							# unique_objects_used_indices = np.append(unique_objects_used_indices, i_object)

							plt.scatter(good_project_planar_coords[0,indices_this_object_of_interest], good_project_planar_coords[1,indices_this_object_of_interest], c = colors[i_object], marker='.')
						plt.xlim((-1, 1))
						plt.ylim((-1, 0))
						fig2.set_size_inches((22, 11))
						ax = plt.gca()
						ax.get_xaxis().set_visible(False)
						ax.axis('off')
						# ax.legend(unique_objects[unique_objects_used_indices.astype(int)])
						plt.savefig('Figures/smarteyeGaze_leftseat'+scenarios[i_scenario]+'.jpg')
						plt.close()
					if (i_seat == 1):
						fig3 = plt.figure(3)

						objects_of_interest = np.array(['Inst_Panel_Capt', 'Inst_Panel_FO',
				       'EFB', 'CDU_Panel_Capt', 'CDU_Panel_FO', 'MCP_Capt',
				       'MCP_FO', 'OutTheWindow','OverheadPanel'])

						unique_objects_used_indices = np.zeros([0])
						for i_object in range(len(objects_of_interest)):
							if (objects_of_interest[i_object] == 'Inst_Panel_Capt'):
								indices_this_object_of_interest = np.where( (object_data_good == 'Inst Panel.KNEE_PANEL_CAPT') | (object_data_good == 'Inst Panel.FIMD_Capt')
								 | (object_data_good == 'Inst Panel.PFD_Capt') | (object_data_good == 'Inst Panel.PFD_Capt_Airspd') | (object_data_good == 'Inst Panel.PFD_Capt_ADI') | (object_data_good == 'Inst Panel.PFD_Capt_Altitude')
								 | (object_data_good == 'Inst Panel.PFD_Capt_FMA') | (object_data_good == 'Inst Panel.Nav_Capt') | (object_data_good == 'Inst Panel.UPPER_EICAS'))
							elif (objects_of_interest[i_object] == 'Inst_Panel_FO'):
								indices_this_object_of_interest = np.where((object_data_good == 'Inst Panel') | (object_data_good == 'Inst Panel.KNEE_PANEL_FO') | (object_data_good == 'Inst Panel.Nav_FO') | (object_data_good == 'Inst Panel.PFD_FO') | (object_data_good == 'Inst Panel.FIMD_FO'))
							elif (objects_of_interest[i_object] == 'EFB_Capt'):
								indices_this_object_of_interest = np.where((object_data_good == 'EFB_Pilot'))
							elif (objects_of_interest[i_object] == 'CDU_Panel_Capt'):
								indices_this_object_of_interest = np.where( (object_data_good == 'CDU_Panel_Capt.CDU_screen_Capt') | (object_data_good == 'CDU_Panel_Capt.CDU_Capt') | (object_data_good == 'CDU_Panel_Capt.FMC_Capt')| (object_data_good == 'CDU_Panel_Capt.MFD'))
							elif (objects_of_interest[i_object] == 'CDU_Panel_FO'):
								indices_this_object_of_interest = np.where((object_data_good == 'CDU_Panel') | (object_data_good == 'CDU_Panel_Capt.CDU_screen_FO') | (object_data_good == 'CDU_Panel_Capt.CDU_FO') )
							elif (objects_of_interest[i_object] == 'MCP_Capt'):
								indices_this_object_of_interest = np.where((object_data_good == 'MCP.EDCP_Capt'))
							elif (objects_of_interest[i_object] == 'MCP_FO'):
								indices_this_object_of_interest = np.where((object_data_good == 'MCP') | (object_data_good == 'MCP.EDCP_FO'))	
							elif (objects_of_interest[i_object] == 'OutTheWindow'):
								indices_this_object_of_interest = np.where((object_data_good == 'OutTheWindow'))
							elif (objects_of_interest[i_object] == 'OverheadPanel'):
								indices_this_object_of_interest = np.where((object_data_good == 'OverheadPanel'))	

							# unique_objects_used_indices = np.append(unique_objects_used_indices, i_unique_object)
							# indices_this_object = np.where(object_data_good == unique_objects[i_unique_object])
							plt.scatter(good_project_planar_coords[0,indices_this_object_of_interest], good_project_planar_coords[1,indices_this_object_of_interest], c = colors[i_object], marker='.')
						plt.xlim((-1, 1))
						plt.ylim((-1, 0))
						fig3.set_size_inches((22, 11))
						ax = plt.gca()
						ax.get_xaxis().set_visible(False)
						ax.axis('off')
						# ax.legend(objects_of_interest)
						plt.savefig('Figures/smarteyeGaze_rightseat'+scenarios[i_scenario]+'.jpg')
						plt.close()
					
					
					# then create another fig that doesn't separate 

					fig1 = plt.figure(1)
					if (i_seat == 0):
						plt.scatter(good_project_planar_coords[0,:], good_project_planar_coords[1,:], c = 'c', alpha=.05,marker='.' )
						plt.text(-.9,-.05,"Gaze Variance="+ str(np.round(total_gaze_variance,3)),c='c', fontsize='large')
						plt.text(-.9,-.10,"Gaze Velocity=" + str(int(np.round(total_average_gaze_velocity))) + "$\pm$" + str(int(np.round(total_std_gaze_velocity)))+"$^\circ$/sec",c='c', fontsize='large')
						# plt.text(-.9,-.1,"%D="+str(int(np.round((good_indices.shape[1] / direction_gaze.shape[1])*100)))+"%; " + 
					else:
						plt.scatter(good_project_planar_coords[0,:], good_project_planar_coords[1,:], c = 'r', alpha=.05,marker='.' )
						plt.text(.65,-.05,"Gaze Variance="+ str(np.round(total_gaze_variance,3)),c='r', fontsize='large')
						plt.text(.65,-.10,"Gaze Velocity=" + str(int(np.round(total_average_gaze_velocity))) + "$\pm$" + str(int(np.round(total_std_gaze_velocity)))+"$^\circ$/sec",c='r', fontsize='large')
					del projected_planar_coords
					del good_indices
					del direction_gaze
					del quality_gaze
					del magnitude

					total_gaze_variance_matrix[i_seat, i_scenario] = total_gaze_variance
					total_gaze_velocity_avg_matrix[i_seat, i_scenario] = total_average_gaze_velocity
					total_gaze_velocity_std_matrix[i_seat, i_scenario] = total_std_gaze_velocity

	plt.xlim((-1, 1))
	plt.ylim((-1, 0))
	# fig1.set_size_inches((22, 11))
	ax = plt.gca()
	ax.get_xaxis().set_visible(False)
	ax.axis('off')
	plt.savefig('Figures/smarteyeGaze_'+scenarios[i_scenario]+'.jpg')
	# plt.show()
	# matplotlib.pyplot.close()
	plt.close()
	np.save("Processing/" + 'gaze_variance_matrix',total_gaze_variance_matrix)
	np.save("Processing/" + 'gaze_velocity_avg_matrix',total_gaze_velocity_avg_matrix)
	np.save("Processing/" + 'gaze_velocity_std_matrix',total_gaze_velocity_std_matrix)
	np.save("Processing/" + 'event_smarteyeGaze_metrics', event_smarteyeGaze_metrics)
	event_smarteyeGazeTimeSeries_metrics.to_csv("Processing/" + 'event_smarteyeGazeTimeSeries_metrics.csv')
	subprocess.call('gsutil -m rsync -r Figures/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Figures"', shell=True)
	subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Processing"', shell=True)


