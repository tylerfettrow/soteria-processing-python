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

def sphere_stereograph(p):
	p = np.squeeze(direction_gaze[:,good_indices])

	[m,n]=p.shape

	s = np.divide(2.0, ( 1.0 + p[m-1,0:n] ))
	ss = np.matlib.repmat( s, m, 1 )

	f = np.zeros((m,1))
	f[m-1] = -1.0
	ff = np.matlib.repmat( f, 1, n )

	q = np.multiply(ss, p) + np.multiply(( 1.0 - ss ), ff);

	b = q[0:2,:]

	return b


def angle_diff(time,direction_gaze):
	degree_per_sec_vector = np.zeros(direction_gaze.shape[1])

	time_diff = np.diff(time)
	for this_frame in range(direction_gaze.shape[1]-1):
		degree_vector = math.degrees(2 * math.atan(la.norm(np.multiply(direction_gaze[:,this_frame],la.norm(direction_gaze[:,this_frame+1])) - np.multiply(la.norm(direction_gaze[:,this_frame]),direction_gaze[:,this_frame+1])) / la.norm(np.multiply(direction_gaze[:,this_frame], la.norm(direction_gaze[:,this_frame+1])) + np.multiply(la.norm(direction_gaze[:,this_frame]), direction_gaze[:,this_frame+1]))))
		degree_per_sec_vector[this_frame+1] = degree_vector / time_diff[this_frame]
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

crews_to_process = ['Crew_01']
file_types = ["smarteye_leftseat", "smarteye_rightseat"]
scenarios = ["1","2","3","5","6","7"]

storage_client = storage.Client(project="soteria-fa59")
bucket = storage_client.bucket("soteria_study_data")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

plt.style.use('dark_background')

for i_crew in range(len(crews_to_process)):
	process_dir_name = crews_to_process[i_crew] + "/Processing/"
	
	gaze_variance_matrix = np.zeros((len(file_types),len(scenarios)))
	gaze_velocity_avg_matrix = np.zeros((len(file_types),len(scenarios)))
	gaze_velocity_std_matrix = np.zeros((len(file_types),len(scenarios)))
	event_smarteyeSphere_metrics = np.zeros((3,3,len(file_types),len(scenarios)))

	f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.npy', 'rb')
	this_event_data = np.load(io.BytesIO(f_stream.read()))

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
		for i_seat in range(len(file_types)):
			blob = bucket.blob(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
			if blob.exists():
				print("QA checking: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				smarteye_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
				object_data = smarteye_data.ObjectIntersectionName.values

				time = np.array(smarteye_data.UserTimeStamp[2:])
				
				# need to find the indices 1 minute before and 1 minute after (start and end of event epoch)
				event1_epoch_start = this_event_data[0, i_scenario] - 60
				difference_array = np.absolute(time-event1_epoch_start)
				event1_start_index = difference_array.argmin()
				event1_epoch_end = this_event_data[0, i_scenario] + 60
				difference_array = np.absolute(time-event1_epoch_end)
				event1_end_index = difference_array.argmin()
				event2_epoch_start = this_event_data[1, i_scenario] - 60
				difference_array = np.absolute(time-event2_epoch_start)
				event2_start_index = difference_array.argmin()
				event2_epoch_end = this_event_data[1, i_scenario] + 60
				difference_array = np.absolute(time-event2_epoch_end)
				event2_end_index = difference_array.argmin()
				
				
				direction_gaze = np.array([smarteye_data.GazeDirectionX[2:], smarteye_data.GazeDirectionY[2:], smarteye_data.GazeDirectionZ[2:]])
				magnitude = np.divide(np.sqrt(np.power(direction_gaze[0,:],2) + np.power(direction_gaze[1,:],2) + np.power(direction_gaze[2,:],2)),1)
				quality_gaze = smarteye_data.GazeDirectionQ[2:]
				degree_per_sec_vector = angle_diff(time, direction_gaze)

				# for the indices that are sequential (i.e. no gaps in good_indices), calculate rate of gaze movement, and remove values that are greater than 700°/s ( Fuchs, A. F. (1967-08-01). "Saccadic and smooth pursuit eye movements in the monkey". The Journal of Physiology. 191 (3): 609–631. doi:10.1113/jphysiol.1967.sp008271. ISSN 1469-7793. PMC 1365495. PMID 4963872.)
				# unit circle 1 -> -1
				# WARNING: not quite what I intended ^^
				good_indices = np.where((magnitude!=0) & (quality_gaze*100 >= 6) & (degree_per_sec_vector<=700))
				good_indices = np.array(good_indices)

				projected_planar_coords = sphere_stereograph(np.squeeze(direction_gaze[:,good_indices]))
				
				object_data_good = np.squeeze(object_data[good_indices])
				unique_objects = np.unique(object_data_good.tolist())

				good_vel_vals = np.squeeze(degree_per_sec_vector[good_indices]) 
				# exclude the first since that was set to 0 on purpose
				average_gaze_velocity = np.average(good_vel_vals[1:])
				std_gaze_velocity = np.std(good_vel_vals[1:])


				# https://www.geeksforgeeks.org/variance-standard-deviation-matrix/
				m = mean(projected_planar_coords,2)
				gaze_variance = variance(projected_planar_coords, 2, m)
				
				# for the event_smarteyeSphere.npy ... and if scenario 7, take middle 5 minutes and store in first column
				# gaze_variance - avg 1st 2 min, avg 1st event, avg 2nd event
				# gaze_vel_avg - avg 1st 2 min, avg 1st event, avg 2nd event
				# gaze_vel_std - avg 1st 2 min, avg 1st event, avg 2nd event
				event_smarteyeSphere_metrics[0,0,i_seat,i_scenario] = 

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

						plt.scatter(projected_planar_coords[0,indices_this_object_of_interest], projected_planar_coords[1,indices_this_object_of_interest], c = colors[i_object], marker='.')
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
						plt.scatter(projected_planar_coords[0,indices_this_object_of_interest], projected_planar_coords[1,indices_this_object_of_interest], c = colors[i_object], marker='.')
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
					plt.scatter(projected_planar_coords[0,:], projected_planar_coords[1,:], c = 'c', alpha=.05,marker='.' )
					plt.text(-.9,-.05,"Gaze Variance="+ str(np.round(gaze_variance,3)),c='c', fontsize='large')
					plt.text(-.9,-.10,"Gaze Velocity=" + str(int(np.round(average_gaze_velocity))) + "$\pm$" + str(int(np.round(std_gaze_velocity)))+"$^\circ$/sec",c='c', fontsize='large')
					# plt.text(-.9,-.1,"%D="+str(int(np.round((good_indices.shape[1] / direction_gaze.shape[1])*100)))+"%; " + 
				else:
					plt.scatter(projected_planar_coords[0,:], projected_planar_coords[1,:], c = 'r', alpha=.05,marker='.' )
					plt.text(.65,-.05,"Gaze Variance="+ str(np.round(gaze_variance,3)),c='r', fontsize='large')
					plt.text(.65,-.10,"Gaze Velocity=" + str(int(np.round(average_gaze_velocity))) + "$\pm$" + str(int(np.round(std_gaze_velocity)))+"$^\circ$/sec",c='r', fontsize='large')
				del projected_planar_coords
				del good_indices
				del direction_gaze
				del quality_gaze
				del magnitude

				gaze_variance_matrix[i_seat, i_scenario] = gaze_variance
				gaze_velocity_avg_matrix[i_seat, i_scenario] = average_gaze_velocity
				gaze_velocity_std_matrix[i_seat, i_scenario] = std_gaze_velocity

		plt.xlim((-1, 1))
		plt.ylim((-1, 0))
		fig1.set_size_inches((22, 11))
		ax = plt.gca()
		ax.get_xaxis().set_visible(False)
		ax.axis('off')
		plt.savefig('Figures/smarteyeGaze_'+scenarios[i_scenario]+'.jpg')
		# plt.show()
		# matplotlib.pyplot.close()
		plt.close()
		np.save("Processing/" + 'gaze_variance_matrix',gaze_variance_matrix)
		np.save("Processing/" + 'gaze_velocity_avg_matrix',gaze_velocity_avg_matrix)
		np.save("Processing/" + 'gaze_velocity_std_matrix',gaze_velocity_std_matrix)
		subprocess.call('gsutil -m rsync -r Figures/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Figures"', shell=True)
		subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Processing"', shell=True)


