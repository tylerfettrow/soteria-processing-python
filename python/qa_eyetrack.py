import os
import numpy as np
import pandas as pd
from os.path import exists
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import seaborn as sns; sns.set_theme()

def unique(list1):
  
    # initialize a null list
    unique_list = []
  
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    # for x in unique_list:
    #     print x
    return unique_list

crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04', 'Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = ['Crew_13']
file_types = ["smarteye_leftseat","smarteye_rightseat"]
scenarios = ["1","2","3","6","7","8","9"]
# file_types = ["smarteye_leftseat"]
# scenarios = ["1"]
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'


# leftseat_heatmap = np.zeros((100,100,len(crews_to_process),len(scenarios)))
# rightseat_heatmap = np.zeros((100,100,len(crews_to_process),len(scenarios)))

leftseat_heatmap = np.zeros((100,100,len(scenarios)))
rightseat_heatmap = np.zeros((100,100,len(scenarios)))

for i_crew in range(len(crews_to_process)):
	pct_usable_matrix = np.zeros((len(scenarios),len(file_types)))
	crew_dir = path_to_project + '/' + crews_to_process[i_crew]
	for i_scenario in range(len(scenarios)):
		for i_seat in range(len(file_types)):
			process_dir_name = crew_dir + "/Processing/"
			if exists(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'):
				print("QA checking: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				eyetrack_data = pd.read_table((process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
				# unique_list = unique(eyetrack_data.ObjectIntersectionName)
				# print(unique_list)

				x = eyetrack_data.ObjectIntersectionX * 1000 # m to mm?
				y = eyetrack_data.ObjectIntersectionY * 1000 # m to mm? where is the origin?
				length_this_trial = eyetrack_data.shape[0]
				data_quality_vector = np.ones(length_this_trial)
				for i_index in range(eyetrack_data.shape[0]):
					if eyetrack_data.IntersectionIndex[i_index] == 0 and eyetrack_data.GazeDirectionQ[i_index] < 50:
						x[i_index] = 0 
						y[i_index] = 0
						data_quality_vector[i_index] = 0
				pct_usable_matrix[i_scenario,i_seat] = (np.sum(data_quality_vector)/length_this_trial) * 100
				x[ x==0 ] = np.nan
				y[ y==0 ] = np.nan
				x = x[~np.isnan(x)]
				y = y[~np.isnan(y)]
				# x = np.round(x)
				# y = np.round(y)
				if file_types[i_seat] == "smarteye_leftseat":
					# plt.hist2d(x, y,bins=100, cmap='jet', range=[[0,1000],[0,400]])
					heatmap, xedges, yedges = np.histogram2d(x, y,bins=100, range=[[0,2000],[0,400]], density=1)
					leftseat_heatmap[:,:,i_scenario] = heatmap.T
					extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

					## Gaze Fig ##
					# plt.imshow(heatmap.T, extent=extent, origin='lower')
					# plt.axis('off')
					# # plt.show()
					# plt.savefig(crew_dir + "/Figures/" + 'smarteye_leftseat_'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0)
					# matplotlib.pyplot.close()
					###############

					## QAp Fig ##


					###############
				elif file_types[i_seat] == "smarteye_rightseat":
					# plt.hist2d(x, y,bins=100, cmap='jet', range=[[600,1600],[0,400]])
					heatmap, xedges, yedges = np.histogram2d(x, y,bins=100, range=[[0,2000],[0,400]], density=1)
					rightseat_heatmap[:,:,i_scenario] = heatmap.T
					extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

					## Gaze Fig ##
					# plt.imshow(heatmap.T, extent=extent, origin='lower')
					# plt.axis('off')
					# # plt.show()
					# plt.savefig(crew_dir + "/Figures/" + 'smarteye_rightseat_'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0)
					# matplotlib.pyplot.close()
					###############

					## QAp Fig ##


					###############

			else:
				pct_usable_matrix[i_scenario,i_seat] = np.nan
				empty_heatmap = np.zeros((100,100))
				if file_types[i_seat] == "smarteye_leftseat":
					leftseat_heatmap[:,:,i_scenario] = empty_heatmap
				elif file_types[i_seat] == "smarteye_rightseat":
					rightseat_heatmap[:,:,i_scenario] = empty_heatmap

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
	plt.savefig(crew_dir + "/Figures/" + 'smarteye_pct_usable.jpg')
	matplotlib.pyplot.close()
	np.save(crew_dir + "/Processing/" + 'pct_usable_matrix',pct_usable_matrix)

	np.save(crew_dir + "/Processing/" + 'smarteye_leftseat_heatmap',leftseat_heatmap)
	np.save(crew_dir + "/Processing/" + 'smarteye_rightseat_heatmap',rightseat_heatmap)
	scenarioAvg_leftseat_heatmap = leftseat_heatmap.mean(2)
	scenarioAvg_rightseat_heatmap = rightseat_heatmap.mean(2) 

	plt.imshow(scenarioAvg_leftseat_heatmap, extent=extent, origin='lower')
	plt.axis('off')
	# plt.show()
	plt.savefig(crew_dir + "/Figures/" + 'smarteye_leftseat_scenarioAvg.tif',bbox_inches='tight',pad_inches=0)
	# os.remove(crew_dir + "/Figures/" + 'smarteye_leftseat_scenarioAvg.jpg')
	matplotlib.pyplot.close()
	plt.imshow(scenarioAvg_rightseat_heatmap, extent=extent, origin='lower')
	plt.axis('off')
	# plt.show()
	plt.savefig(crew_dir + "/Figures/" + 'smarteye_rightseat_scenarioAvg.tif',bbox_inches='tight',pad_inches=0)
	# os.remove(crew_dir + "/Figures/" + 'smarteye_rightseat_scenarioAvg.jpg')
	matplotlib.pyplot.close()