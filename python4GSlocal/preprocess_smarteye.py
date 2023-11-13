import os
import numpy as np
import numpy.matlib
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
from numpy import linalg as la
from os.path import exists
from distinctipy import distinctipy
# from tensorflow.python.lib.io import file_io
import io
import math
import statistics
import matplotlib.colors as colors
import shutil
from collections import Counter

########### SETTINGS ##################
# crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
crews_to_process = ['Crew_13']
file_types = ["smarteye_leftseat","smarteye_rightseat"]
scenarios = ["1","2","3","5","6","7"]
plot_qatable = 1 # embeds pct_usable value too
plot_aoi = 0
time_per_epoch_4_analysis = 2
#######################################

def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

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
	ss = numpy.matlib.repmat( s, m, 1 )

	f = np.zeros((m,1))
	f[m-1] = -1.0
	ff = numpy.matlib.repmat( f, 1, n )

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


# scenarios = ["1"]
storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")


leftseat_heatmap = np.zeros((100,100,len(scenarios)))
rightseat_heatmap = np.zeros((100,100,len(scenarios)))

for i_crew in range(len(crews_to_process)):

	if exists("Figures"):
		# subprocess.Popen('rm -rf Figures', shell=True)
		shutil.rmtree('Figures', ignore_errors=True)
		time.sleep(5)
		os.mkdir("Figures")
	else:
		os.mkdir("Figures")
	if exists("Processing"):
		# subprocess.Popen('rm -rf Processing', shell=True)
		shutil.rmtree('Processing', ignore_errors=True)
		time.sleep(5)
		os.mkdir("Processing")
	else:
		os.mkdir("Processing")
		
	pct_usable_matrix = np.zeros((len(scenarios),len(file_types)))
	crew_dir = crews_to_process[i_crew]
	process_dir_name = crew_dir + "/Processing/"

	event_smarteyeGazeTimeSeries_metrics = pd.DataFrame()
	event_smarteyeTimeSeries_metrics = pd.DataFrame()

	# f_stream = io.open('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.npy', 'rb')
	# this_event_data = np.load(io.BytesIO(f_stream.read()))

	# this_event_data = np.load('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.npy')
	this_event_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + 'event_vector_scenario.csv'),delimiter=',')
	this_event_data = np.array(this_event_data)
	this_event_data = this_event_data[:,1:]

	for i_scenario in range(len(scenarios)):
		if ((getCrewInt(crews_to_process[i_crew]) == 13) & (scenarios[i_scenario] == '5')):
			pct_usable_matrix[i_scenario,:] = 0
		else:
			for i_seat in range(len(file_types)):

				blob = bucket.blob(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				if blob.exists():
					smarteye_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
					print("Processing Crew: " + crews_to_process[i_crew] + " Scenario: "+scenarios[i_scenario])
					time_vector = np.array(smarteye_data.UserTimeStamp[2:])
					HeadRotationQ = np.array(smarteye_data.HeadRotationQ[2:])
					PupilDiameterQ = np.array(smarteye_data.PupilDiameterQ[2:])

					direction_gaze = np.array([smarteye_data.GazeDirectionX[2:], smarteye_data.GazeDirectionY[2:], smarteye_data.GazeDirectionZ[2:]])
					HeadPosition = np.array([smarteye_data.HeadPosX[2:], smarteye_data.HeadPosY[2:], smarteye_data.HeadPosZ[2:]])
					magnitude = np.divide(np.sqrt(np.power(direction_gaze[0,:],2) + np.power(direction_gaze[1,:],2) + np.power(direction_gaze[2,:],2)),1)
					quality_gaze = smarteye_data.GazeDirectionQ[2:]
					degree_per_sec_vector = angle_diff(time_vector, direction_gaze)

					# for the indices that are sequential (i.e. no gaps in good_indices), calculate rate of gaze movement, and remove values that are greater than 700°/s ( Fuchs, A. F. (1967-08-01). "Saccadic and smooth pursuit eye movements in the monkey". The Journal of Physiology. 191 (3): 609–631. doi:10.1113/jphysiol.1967.sp008271. ISSN 1469-7793. PMC 1365495. PMID 4963872.)
					# unit circle 1 -> -1
					# WARNING: not quite what I intended ^^
					good_indices = np.squeeze(np.where((magnitude!=0) & (quality_gaze*100 >= 6) & (degree_per_sec_vector<=700) & (degree_per_sec_vector!=np.nan)))

					length_this_trial = smarteye_data.shape[0]
					pct_usable_matrix[i_scenario,i_seat] = np.rint((len(good_indices)/length_this_trial) * 100)

					# projected_planar_coords = sphere_stereograph(np.squeeze(direction_gaze[:,good_indices]))
					projected_headPos_coords = sphere_stereograph(np.squeeze(HeadPosition))
					projected_planar_coords = sphere_stereograph(np.squeeze(direction_gaze))
					good_project_planar_coords = projected_planar_coords[:,good_indices]
					good_project_headPos_coords = projected_headPos_coords[:,good_indices]
					
					good_project_planar_coords = good_project_planar_coords - good_project_headPos_coords # correct origin of gaze vector for head position
					good_project_planar_coords[0,:] = good_project_planar_coords[0,:] * -1 # flip the x axis of gaze vector

					good_ObjectIntersection = smarteye_data.ObjectIntersectionName[good_indices]
					# object_data_good = np.squeeze(object_data[good_indices])
					# unique_objects = np.unique(object_data_good.tolist())

					total_good_vel_vals = np.squeeze(degree_per_sec_vector[good_indices])
					# exclude the first since that was set to 0 on purpose
					total_average_gaze_velocity = np.average(total_good_vel_vals[1:])
					total_std_gaze_velocity = np.std(total_good_vel_vals[1:])

					# https://www.geeksforgeeks.org/variance-standard-deviation-matrix/
					total_m = mean(projected_planar_coords[:,good_indices],2)
					total_gaze_variance = variance(projected_planar_coords[:,good_indices], 2, total_m)

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

					number_of_epochs_this_scenario = np.floor(time_vector[-1]/time_per_epoch_4_analysis)
					this_smarteyeTimeSeries_np = np.zeros((int(number_of_epochs_this_scenario), 9))
					this_smarteyeTimeSeries_np[:,0] = getCrewInt(crews_to_process[i_crew])
					this_smarteyeGazeTimeSeries_np = np.zeros((int(number_of_epochs_this_scenario), 9))
					this_smarteyeGazeTimeSeries_np[:,0] = getCrewInt(crews_to_process[i_crew])
					if (i_seat == 0):
						this_smarteyeTimeSeries_np[:,1] = 0
						this_smarteyeGazeTimeSeries_np[:,1] = 0
					else:
						this_smarteyeTimeSeries_np[:,1] = 1
						this_smarteyeGazeTimeSeries_np[:,1] = 1
				
					this_smarteyeTimeSeries_np[:,2] = i_scenario
					length_this_data = smarteye_data.shape[0]
					for this_epoch in range(int(number_of_epochs_this_scenario)):
						this_epoch_indices_start = np.floor(length_this_data/number_of_epochs_this_scenario) * this_epoch
						this_epoch_indices_end = this_epoch_indices_start + np.floor(length_this_data/number_of_epochs_this_scenario)

						if ((time_vector[int(this_epoch_indices_start)] > this_event_data[0, i_scenario] - 60) & (time_vector[int(this_epoch_indices_start)] < this_event_data[0, i_scenario] + 60)) | ((time_vector[int(this_epoch_indices_start)] > this_event_data[1, i_scenario] - 60) & (time_vector[int(this_epoch_indices_start)] < this_event_data[1, i_scenario] + 60)):
							this_smarteyeTimeSeries_np[this_epoch, 3] = 1
							this_smarteyeGazeTimeSeries_np[this_epoch, 3] = 1
						else:
							this_smarteyeTimeSeries_np[this_epoch, 3] = 0
							this_smarteyeGazeTimeSeries_np[this_epoch, 3] = 0

						this_smarteyeGazeTimeSeries_np[this_epoch, 4] = this_epoch
						this_good_indices = np.squeeze(np.where((good_indices > this_epoch_indices_start) & (good_indices < this_epoch_indices_end)))
						if (this_good_indices.size > 1):
							this_smarteyeGazeTimeSeries_np[this_epoch, 5] = variance(good_project_planar_coords[:,this_good_indices], 2, mean(good_project_planar_coords[:,this_good_indices],1))
							this_smarteyeGazeTimeSeries_np[this_epoch, 6] = np.nanmean(np.squeeze(degree_per_sec_vector[this_good_indices]).T)
							this_smarteyeGazeTimeSeries_np[this_epoch, 7] = np.nanstd(np.squeeze(degree_per_sec_vector[this_good_indices]).T)

							unique_ObjectIntersection  = unique(good_ObjectIntersection)
							for i in range(len(unique_ObjectIntersection)):
							    indices_this_object = np.squeeze(np.where(good_ObjectIntersection == unique_ObjectIntersection[i]))
							    
							    # col= cm(1.*i/NUM_COLORS)
							    # ax.scatter(good_project_planar_coords[0,indices_this_object], good_project_planar_coords[1,indices_this_object], color = col, alpha=.5,marker='.' )
							    # ax.legend(unique_ObjectIntersection)
							indices_this_object = np.squeeze(np.where(good_ObjectIntersection == unique_ObjectIntersection[i]))
							# good_ObjectIntersection_df = np.array(pd.DataFrame(good_ObjectIntersection))
							# 	good_ObjectIntersection_df[pd.Index(this_good_indices)]
							# good_ObjectIntersection[pd.Index(this_good_indices)]
							x = Counter(good_ObjectIntersection[good_ObjectIntersection.index[this_good_indices]])

							this_smarteyeGazeTimeSeries_np[this_epoch, 8] = x['MCP']
						else:
							this_smarteyeGazeTimeSeries_np[this_epoch, 5] = np.nan
							this_smarteyeGazeTimeSeries_np[this_epoch, 6] = np.nan
							this_smarteyeGazeTimeSeries_np[this_epoch, 7] = np.nan

						this_smarteyeTimeSeries_np[this_epoch, 4] = this_epoch
						this_smarteyeTimeSeries_np[this_epoch, 5] = np.nanmean(headheadingDeg_rate[int(this_epoch_indices_start):int(this_epoch_indices_end)].T)
						this_smarteyeTimeSeries_np[this_epoch, 6] = np.nanstd(headheadingDeg_rate[int(this_epoch_indices_start):int(this_epoch_indices_end)].T)
						this_smarteyeTimeSeries_np[this_epoch, 7] = np.nanmean(pupilD[int(this_epoch_indices_start):int(this_epoch_indices_end)].T)
						this_smarteyeTimeSeries_np[this_epoch, 8] = np.nanstd(pupilD[int(this_epoch_indices_start):int(this_epoch_indices_end)].T)

					this_smarteyeGazeTimeSeries_df = pd.DataFrame(this_smarteyeGazeTimeSeries_np)
					this_smarteyeGazeTimeSeries_df.columns = ['crew', 'seat', 'scenario', 'event_label', 'epoch_index', 'gaze_variance', 'gaze_vel_avg', 'gaze_vel_std', 'AOI']
					event_smarteyeGazeTimeSeries_metrics = pd.concat([event_smarteyeGazeTimeSeries_metrics,this_smarteyeGazeTimeSeries_df])

					this_smarteyeTimeSeries_df = pd.DataFrame(this_smarteyeTimeSeries_np)
					this_smarteyeTimeSeries_df.columns = ['crew', 'seat', 'scenario', 'event_label', 'epoch_index','headHeading_avg', 'headHeading_std', 'pupilD_avg', 'pupilD_std']
					event_smarteyeTimeSeries_metrics = pd.concat([event_smarteyeTimeSeries_metrics,this_smarteyeTimeSeries_df])

				# assign colors to each possible object (i.e. make a map for each object)
				# for each object, determine what indices are labeled with it, then plot those indices with the mapped color

				# for i_obj in range(len(unique_ObjectIntersection)):
				# 	"Inst Panel" in good_ObjectIntersection

					if plot_aoi:
						NUM_COLORS = len(unique_ObjectIntersection)
						cm = plt.get_cmap('gist_rainbow')
						cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)

						fig1 = plt.figure(1)
						ax = plt.gca()
						# color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
						
						if (i_seat == 0):
							for i in range(NUM_COLORS):
							    indices_this_object = np.squeeze(np.where(good_ObjectIntersection == unique_ObjectIntersection[i]))
							    col= cm(1.*i/NUM_COLORS)
							    ax.scatter(good_project_planar_coords[0,indices_this_object], good_project_planar_coords[1,indices_this_object], color = col, alpha=.5,marker='.' )
							    ax.legend(unique_ObjectIntersection)
							    ax.set_title('Gaze vector left seat / Capt')
							
							fig1.set_size_inches((22, 11))
							ax = plt.gca()
							ax.get_xaxis().set_visible(False)
							ax.axis('off')
							plt.savefig('Figures/smarteyeGaze_'+scenarios[i_scenario]+'_leftseat.jpg')
							plt.close()
						else:
							for i in range(NUM_COLORS):
							    indices_this_object = np.squeeze(np.where(good_ObjectIntersection == unique_ObjectIntersection[i]))
							    col= cm(1.*i/NUM_COLORS)
							    ax.scatter(good_project_planar_coords[0,indices_this_object], good_project_planar_coords[1,indices_this_object], color = col, alpha=.5,marker='.' )
							    ax.legend(unique_ObjectIntersection)
							    ax.set_title('Gaze vector right seat / FO')

							fig1.set_size_inches((22, 11))
							ax = plt.gca()
							ax.get_xaxis().set_visible(False)
							ax.axis('off')
							plt.savefig('Figures/smarteyeGaze_'+scenarios[i_scenario]+'_rightseat.jpg')
							plt.close()

	if plot_qatable:
		fig, ax = plt.subplots()
		cbar_kws = { 'ticks' : [0, 100] }
		ax = sns.heatmap(pct_usable_matrix, linewidths=.5, cbar_kws = cbar_kws,annot=True,fmt='.3g')
		ax.set_xticklabels(file_types)
		ax.set_yticklabels(scenarios)
		ax.set(xlabel='pilot', ylabel='scenarios')
		plt.yticks(rotation=0)
		ax.xaxis.set_label_position('top')
		ax.xaxis.tick_top()
		fig.tight_layout()
		# plt.show()
		plt.savefig("Figures/" + 'smarteye_pct_usable.jpg')
		matplotlib.pyplot.close()
		np.save("Processing/" + 'smarteye_pct_usable_matrix',pct_usable_matrix)

	pct_usable_matrix_df = pd.DataFrame(pct_usable_matrix)
	pct_usable_matrix_df.to_csv("Processing/" + 'smarteye_pct_usable_matrix.csv')
	event_smarteyeGazeTimeSeries_metrics.to_csv("Processing/" + 'event_smarteyeGazeTimeSeries_metrics.csv')
	event_smarteyeTimeSeries_metrics.to_csv("Processing/" + 'event_smarteyeTimeSeries_metrics.csv')
	subprocess.call('gsutil -m rsync -r Figures/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Figures"', shell=True)
	subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Processing"', shell=True)