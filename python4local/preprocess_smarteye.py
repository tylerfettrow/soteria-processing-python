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

crews_to_process = ['Crew_01']
file_types = ["smarteye_leftseat","smarteye_rightseat"]
scenarios = ["1","2","3","5","6","7","8","9"]

path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'

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
plot_realtime = 1
vertical_mm = 1575
horizontal_mm = 400
number_of_epochs = 1000

for i_crew in range(len(crews_to_process)):
	pct_usable_matrix = np.zeros((len(scenarios),len(file_types)))
	crew_dir = path_to_project + '/' + crews_to_process[i_crew]
	process_dir_name = crew_dir + "/Processing/"

	pupild_leftseat = np.zeros((number_of_epochs,len(scenarios)))
	pupild_rightseat = np.zeros((number_of_epochs,len(scenarios)))
	headHeading_leftseat = np.zeros((number_of_epochs,len(scenarios)))
	headHeading_rightseat = np.zeros((number_of_epochs,len(scenarios)))
	smarteye_timesec_epoch_storage = np.zeros((len(scenarios),number_of_epochs))

	for i_scenario in range(len(scenarios)):
		for i_seat in range(len(file_types)):
			if exists(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'):
				print("QA checking: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				smarteye_data = pd.read_table((process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
				
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

				if plot_realtime:
					smarteye_data

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
					
					if file_types[i_seat] == "smarteye_leftseat":
						heatmap, xedges, yedges = np.histogram2d(x, y,bins=100, range=[[0,vertical_mm],[0,horizontal_mm]], density=1)
						leftseat_heatmap[:,:,i_scenario] = heatmap.T
						extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

						ax = plt.subplot(len(scenarios), 2, i_scenario*2 + 1)
		
						## Gaze Fig ##
						plt.imshow(heatmap.T, extent=extent, origin='lower')
						# plt.axis('off')
						plt.ylabel(scenarios[i_scenario], rotation=0)
						ax.yaxis.label.set_color('white')
						# plt.yticks(rotation=90)
						if i_scenario == 0:
							plt.xlabel('leftseat')
							ax.xaxis.set_label_position('top')
							ax.xaxis.label.set_color('white')
						ax.get_xaxis().set_ticks([])
						ax.get_yaxis().set_ticks([])
						ax.spines['bottom'].set_color('black')
						ax.spines['top'].set_color('black')
						ax.spines['right'].set_color('black')
						ax.spines['left'].set_color('black')
						ax.text(0.9, 0.9, pct_usable_matrix[i_scenario,i_seat], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,color='white',fontsize='x-small')
						# df['leftseat_'+scenarios[i_scenario]] = heatmap.T.tolist()

						# PFD_CAPT
						plt.axhline(y = PFD_capt_lowerleft[1], xmin = PFD_capt_lowerleft[0]/vertical_mm, xmax = (PFD_capt_lowerleft[0] + PFD_capt_widthheight[0])/vertical_mm, color = 'w',linestyle='-')
						plt.axhline(y = PFD_capt_lowerleft[1]+ PFD_capt_widthheight[1], xmin = PFD_capt_lowerleft[0]/vertical_mm, xmax = (PFD_capt_lowerleft[0] + PFD_capt_widthheight[0])/vertical_mm, color = 'w',linestyle='-')
						plt.axvline(x = PFD_capt_lowerleft[0], ymin = PFD_capt_lowerleft[1]/horizontal_mm, ymax = (PFD_capt_lowerleft[1] + PFD_capt_widthheight[1])/horizontal_mm, color = 'w',linestyle='-')
						plt.axvline(x = PFD_capt_lowerleft[0]+ PFD_capt_widthheight[0], ymin = PFD_capt_lowerleft[1]/horizontal_mm, ymax = (PFD_capt_lowerleft[1] + PFD_capt_widthheight[1])/horizontal_mm, color = 'w',linestyle='-')

						# kneePanel
						# plt.axhline(y = kneePanel_capt_lowerleft[1], xmin = kneePanel_capt_lowerleft[0]/2000, xmax = (kneePanel_capt_lowerleft[0] + kneePanel_capt_widthheight[0])/2000, color = 'b',linestyle='--')
						# plt.axhline(y = kneePanel_capt_lowerleft[1]+ kneePanel_capt_widthheight[0], xmin = kneePanel_capt_lowerleft[0]/2000, xmax = (kneePanel_capt_lowerleft[0] + kneePanel_capt_widthheight[0])/2000, color = 'b',linestyle='--')
						# plt.axvline(x = kneePanel_capt_lowerleft[0], ymin = kneePanel_capt_lowerleft[1]/horizontal_mm, ymax = (kneePanel_capt_lowerleft[1] + kneePanel_capt_widthheight[1])/horizontal_mm, color = 'b',linestyle='--')
						# plt.axvline(x = kneePanel_capt_lowerleft[0]+ kneePanel_capt_widthheight[1], ymin = kneePanel_capt_lowerleft[1]/horizontal_mm, ymax = (kneePanel_capt_lowerleft[1] + kneePanel_capt_widthheight[1])/horizontal_mm, color = 'b',linestyle='--')

						# FIMD_CAPT
						plt.axhline(y = FIMD_capt_lowerleft[1], xmin = FIMD_capt_lowerleft[0]/vertical_mm, xmax = (FIMD_capt_lowerleft[0] + FIMD_capt_widthheight[0])/vertical_mm, color = 'r',linestyle='-')
						plt.axhline(y = FIMD_capt_lowerleft[1]+ FIMD_capt_widthheight[1], xmin = FIMD_capt_lowerleft[0]/vertical_mm, xmax = (FIMD_capt_lowerleft[0] + FIMD_capt_widthheight[0])/vertical_mm, color = 'r',linestyle='-')
						plt.axvline(x = FIMD_capt_lowerleft[0], ymin = FIMD_capt_lowerleft[1]/horizontal_mm, ymax = (FIMD_capt_lowerleft[1] + FIMD_capt_widthheight[1])/horizontal_mm, color = 'r',linestyle='-')
						plt.axvline(x = FIMD_capt_lowerleft[0]+ FIMD_capt_widthheight[0], ymin = FIMD_capt_lowerleft[1]/horizontal_mm, ymax = (FIMD_capt_lowerleft[1] + FIMD_capt_widthheight[1])/horizontal_mm, color = 'r',linestyle='-')

						# NAV_CAPT
						plt.axhline(y = Nav_capt_lowerleft[1], xmin = Nav_capt_lowerleft[0]/vertical_mm, xmax = (Nav_capt_lowerleft[0] + Nav_capt_width_height[0])/vertical_mm, color = 'b',linestyle='-')
						plt.axhline(y = Nav_capt_lowerleft[1]+ Nav_capt_width_height[1], xmin = Nav_capt_lowerleft[0]/vertical_mm, xmax = (Nav_capt_lowerleft[0] + Nav_capt_width_height[0])/vertical_mm, color = 'b',linestyle='-')
						plt.axvline(x = Nav_capt_lowerleft[0], ymin = Nav_capt_lowerleft[1]/horizontal_mm, ymax = (Nav_capt_lowerleft[1] + Nav_capt_width_height[1])/horizontal_mm, color = 'b',linestyle='-')
						plt.axvline(x = Nav_capt_lowerleft[0]+ Nav_capt_width_height[0], ymin = Nav_capt_lowerleft[1]/horizontal_mm, ymax = (Nav_capt_lowerleft[1] + Nav_capt_width_height[1])/horizontal_mm, color = 'b',linestyle='-')

						# UpperEICAS
						plt.axhline(y = UpperEICAS_capt_lowerleft[1], xmin = UpperEICAS_capt_lowerleft[0]/vertical_mm, xmax = (UpperEICAS_capt_lowerleft[0] + UpperEICAS_capt_widthheight[0])/vertical_mm, color = 'm',linestyle='-')
						plt.axhline(y = UpperEICAS_capt_lowerleft[1]+ UpperEICAS_capt_widthheight[1], xmin = UpperEICAS_capt_lowerleft[0]/vertical_mm, xmax = (UpperEICAS_capt_lowerleft[0] + UpperEICAS_capt_widthheight[0])/vertical_mm, color = 'm',linestyle='-')
						plt.axvline(x = UpperEICAS_capt_lowerleft[0], ymin = UpperEICAS_capt_lowerleft[1]/horizontal_mm, ymax = (UpperEICAS_capt_lowerleft[1] + UpperEICAS_capt_widthheight[1])/horizontal_mm, color = 'm',linestyle='-')
						plt.axvline(x = UpperEICAS_capt_lowerleft[0]+ UpperEICAS_capt_widthheight[0], ymin = UpperEICAS_capt_lowerleft[1]/horizontal_mm, ymax = (UpperEICAS_capt_lowerleft[1] + UpperEICAS_capt_widthheight[1])/horizontal_mm, color = 'm',linestyle='-')

						# NAV_FO
						plt.axhline(y = Nav_fo_lowerleft[1], xmin = Nav_fo_lowerleft[0]/vertical_mm, xmax = (Nav_fo_lowerleft[0] + Nav_fo_widthheight[0])/vertical_mm, color = 'b',linestyle='--')
						plt.axhline(y = Nav_fo_lowerleft[1]+ Nav_fo_widthheight[1], xmin = Nav_fo_lowerleft[0]/vertical_mm, xmax = (Nav_fo_lowerleft[0] + Nav_fo_widthheight[0])/vertical_mm, color = 'b',linestyle='--')
						plt.axvline(x = Nav_fo_lowerleft[0], ymin = Nav_fo_lowerleft[1]/horizontal_mm, ymax = (Nav_fo_lowerleft[1] + Nav_fo_widthheight[1])/horizontal_mm, color = 'b',linestyle='--')
						plt.axvline(x = Nav_fo_lowerleft[0]+ Nav_fo_widthheight[0], ymin = Nav_fo_lowerleft[1]/horizontal_mm, ymax = (Nav_fo_lowerleft[1] + Nav_fo_widthheight[1])/horizontal_mm, color = 'b',linestyle='--')

						# PFD_FO
						plt.axhline(y = PFD_fo_lowerleft[1], xmin = PFD_fo_lowerleft[0]/vertical_mm, xmax = (PFD_fo_lowerleft[0] + PFD_fo_widthheight[0])/vertical_mm, color = 'w',linestyle='--')
						plt.axhline(y = PFD_fo_lowerleft[1]+ PFD_fo_widthheight[1], xmin = PFD_fo_lowerleft[0]/vertical_mm, xmax = (PFD_fo_lowerleft[0] + PFD_fo_widthheight[0])/vertical_mm, color = 'w',linestyle='--')
						plt.axvline(x = PFD_fo_lowerleft[0], ymin = PFD_fo_lowerleft[1]/horizontal_mm, ymax = (PFD_fo_lowerleft[1] + PFD_fo_widthheight[1])/horizontal_mm, color = 'w',linestyle='--')
						plt.axvline(x = PFD_fo_lowerleft[0]+ PFD_fo_widthheight[0], ymin = PFD_fo_lowerleft[1]/horizontal_mm, ymax = (PFD_fo_lowerleft[1] + PFD_fo_widthheight[1])/horizontal_mm, color = 'w',linestyle='--')

						# FIMD_FO
						plt.axhline(y = FIMD_fo_lowerleft[1], xmin = FIMD_fo_lowerleft[0]/vertical_mm, xmax = (FIMD_fo_lowerleft[0] + FIMD_fo_widthheight[0])/vertical_mm, color = 'r',linestyle='--')
						plt.axhline(y = FIMD_fo_lowerleft[1]+ FIMD_fo_widthheight[1], xmin = FIMD_fo_lowerleft[0]/vertical_mm, xmax = (FIMD_fo_lowerleft[0] + FIMD_fo_widthheight[0])/vertical_mm, color = 'r',linestyle='--')
						plt.axvline(x = FIMD_fo_lowerleft[0], ymin = FIMD_fo_lowerleft[1]/horizontal_mm, ymax = (FIMD_fo_lowerleft[1] + FIMD_fo_widthheight[1])/horizontal_mm, color = 'r',linestyle='--')
						plt.axvline(x = FIMD_fo_lowerleft[0]+ FIMD_fo_widthheight[0], ymin = FIMD_fo_lowerleft[1]/horizontal_mm, ymax = (FIMD_fo_lowerleft[1] + FIMD_fo_widthheight[1])/horizontal_mm, color = 'r',linestyle='--')

						# plt.show()
						# plt.savefig(crew_dir + "/Figures/" + 'smarteye_leftseat_'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0)
						#matplotlib.pyplot.close()
						###############

						## QAp Fig ##


						###############
					elif file_types[i_seat] == "smarteye_rightseat":
						# plt.hist2d(x, y,bins=100, cmap='jet', range=[[600,1600],[0,horizontal_mm]])
						heatmap, xedges, yedges = np.histogram2d(x, y,bins=100, range=[[0,vertical_mm],[0,horizontal_mm]], density=1)
						rightseat_heatmap[:,:,i_scenario] = heatmap.T
						extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


						ax = plt.subplot(len(scenarios), 2, i_scenario * 2 + 2)
						# ax.set_title('rightseat_'+scenarios[i_scenario])
				
						## Gaze Fig ##
						plt.imshow(heatmap.T, extent=extent, origin='lower')
						# plt.axis('off')
						if i_scenario == 0:
							plt.xlabel('rightseat')
							ax.xaxis.set_label_position('top')
							ax.xaxis.label.set_color('white')
						ax.get_xaxis().set_ticks([])
						ax.get_yaxis().set_ticks([])
						ax.spines['bottom'].set_color('black')
						ax.spines['top'].set_color('black') 
						ax.spines['right'].set_color('black')
						ax.spines['left'].set_color('black')
						ax.text(0.9, 0.9, pct_usable_matrix[i_scenario,i_seat], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,color='white',fontsize='x-small')
						# df['leftseat_'+scenarios[i_scenario]] = heatmap.T.tolist()

						# PFD_CAPT
						plt.axhline(y = PFD_capt_lowerleft[1], xmin = PFD_capt_lowerleft[0]/vertical_mm, xmax = (PFD_capt_lowerleft[0] + PFD_capt_widthheight[0])/vertical_mm, color = 'w',linestyle='-')
						plt.axhline(y = PFD_capt_lowerleft[1]+ PFD_capt_widthheight[1], xmin = PFD_capt_lowerleft[0]/vertical_mm, xmax = (PFD_capt_lowerleft[0] + PFD_capt_widthheight[0])/vertical_mm, color = 'w',linestyle='-')
						plt.axvline(x = PFD_capt_lowerleft[0], ymin = PFD_capt_lowerleft[1]/horizontal_mm, ymax = (PFD_capt_lowerleft[1] + PFD_capt_widthheight[1])/horizontal_mm, color = 'w',linestyle='-')
						plt.axvline(x = PFD_capt_lowerleft[0]+ PFD_capt_widthheight[0], ymin = PFD_capt_lowerleft[1]/horizontal_mm, ymax = (PFD_capt_lowerleft[1] + PFD_capt_widthheight[1])/horizontal_mm, color = 'w',linestyle='-')

						# kneePanel
						# plt.axhline(y = kneePanel_capt_lowerleft[1], xmin = kneePanel_capt_lowerleft[0]/2000, xmax = (kneePanel_capt_lowerleft[0] + kneePanel_capt_widthheight[0])/2000, color = 'b',linestyle='--')
						# plt.axhline(y = kneePanel_capt_lowerleft[1]+ kneePanel_capt_widthheight[0], xmin = kneePanel_capt_lowerleft[0]/2000, xmax = (kneePanel_capt_lowerleft[0] + kneePanel_capt_widthheight[0])/2000, color = 'b',linestyle='--')
						# plt.axvline(x = kneePanel_capt_lowerleft[0], ymin = kneePanel_capt_lowerleft[1]/horizontal_mm, ymax = (kneePanel_capt_lowerleft[1] + kneePanel_capt_widthheight[1])/horizontal_mm, color = 'b',linestyle='--')
						# plt.axvline(x = kneePanel_capt_lowerleft[0]+ kneePanel_capt_widthheight[1], ymin = kneePanel_capt_lowerleft[1]/horizontal_mm, ymax = (kneePanel_capt_lowerleft[1] + kneePanel_capt_widthheight[1])/horizontal_mm, color = 'b',linestyle='--')

						# FIMD_CAPT
						plt.axhline(y = FIMD_capt_lowerleft[1], xmin = FIMD_capt_lowerleft[0]/vertical_mm, xmax = (FIMD_capt_lowerleft[0] + FIMD_capt_widthheight[0])/vertical_mm, color = 'r',linestyle='-')
						plt.axhline(y = FIMD_capt_lowerleft[1]+ FIMD_capt_widthheight[1], xmin = FIMD_capt_lowerleft[0]/vertical_mm, xmax = (FIMD_capt_lowerleft[0] + FIMD_capt_widthheight[0])/vertical_mm, color = 'r',linestyle='-')
						plt.axvline(x = FIMD_capt_lowerleft[0], ymin = FIMD_capt_lowerleft[1]/horizontal_mm, ymax = (FIMD_capt_lowerleft[1] + FIMD_capt_widthheight[1])/horizontal_mm, color = 'r',linestyle='-')
						plt.axvline(x = FIMD_capt_lowerleft[0]+ FIMD_capt_widthheight[0], ymin = FIMD_capt_lowerleft[1]/horizontal_mm, ymax = (FIMD_capt_lowerleft[1] + FIMD_capt_widthheight[1])/horizontal_mm, color = 'r',linestyle='-')

						# NAV_CAPT
						plt.axhline(y = Nav_capt_lowerleft[1], xmin = Nav_capt_lowerleft[0]/vertical_mm, xmax = (Nav_capt_lowerleft[0] + Nav_capt_width_height[0])/vertical_mm, color = 'b',linestyle='-')
						plt.axhline(y = Nav_capt_lowerleft[1]+ Nav_capt_width_height[1], xmin = Nav_capt_lowerleft[0]/vertical_mm, xmax = (Nav_capt_lowerleft[0] + Nav_capt_width_height[0])/vertical_mm, color = 'b',linestyle='-')
						plt.axvline(x = Nav_capt_lowerleft[0], ymin = Nav_capt_lowerleft[1]/horizontal_mm, ymax = (Nav_capt_lowerleft[1] + Nav_capt_width_height[1])/horizontal_mm, color = 'b',linestyle='-')
						plt.axvline(x = Nav_capt_lowerleft[0]+ Nav_capt_width_height[0], ymin = Nav_capt_lowerleft[1]/horizontal_mm, ymax = (Nav_capt_lowerleft[1] + Nav_capt_width_height[1])/horizontal_mm, color = 'b',linestyle='-')

						# UpperEICAS
						plt.axhline(y = UpperEICAS_capt_lowerleft[1], xmin = UpperEICAS_capt_lowerleft[0]/vertical_mm, xmax = (UpperEICAS_capt_lowerleft[0] + UpperEICAS_capt_widthheight[0])/vertical_mm, color = 'm',linestyle='-')
						plt.axhline(y = UpperEICAS_capt_lowerleft[1]+ UpperEICAS_capt_widthheight[1], xmin = UpperEICAS_capt_lowerleft[0]/vertical_mm, xmax = (UpperEICAS_capt_lowerleft[0] + UpperEICAS_capt_widthheight[0])/vertical_mm, color = 'm',linestyle='-')
						plt.axvline(x = UpperEICAS_capt_lowerleft[0], ymin = UpperEICAS_capt_lowerleft[1]/horizontal_mm, ymax = (UpperEICAS_capt_lowerleft[1] + UpperEICAS_capt_widthheight[1])/horizontal_mm, color = 'm',linestyle='-')
						plt.axvline(x = UpperEICAS_capt_lowerleft[0]+ UpperEICAS_capt_widthheight[0], ymin = UpperEICAS_capt_lowerleft[1]/horizontal_mm, ymax = (UpperEICAS_capt_lowerleft[1] + UpperEICAS_capt_widthheight[1])/horizontal_mm, color = 'm',linestyle='-')

						# NAV_FO
						plt.axhline(y = Nav_fo_lowerleft[1], xmin = Nav_fo_lowerleft[0]/vertical_mm, xmax = (Nav_fo_lowerleft[0] + Nav_fo_widthheight[0])/vertical_mm, color = 'b',linestyle='--')
						plt.axhline(y = Nav_fo_lowerleft[1]+ Nav_fo_widthheight[1], xmin = Nav_fo_lowerleft[0]/vertical_mm, xmax = (Nav_fo_lowerleft[0] + Nav_fo_widthheight[0])/vertical_mm, color = 'b',linestyle='--')
						plt.axvline(x = Nav_fo_lowerleft[0], ymin = Nav_fo_lowerleft[1]/horizontal_mm, ymax = (Nav_fo_lowerleft[1] + Nav_fo_widthheight[1])/horizontal_mm, color = 'b',linestyle='--')
						plt.axvline(x = Nav_fo_lowerleft[0]+ Nav_fo_widthheight[0], ymin = Nav_fo_lowerleft[1]/horizontal_mm, ymax = (Nav_fo_lowerleft[1] + Nav_fo_widthheight[1])/horizontal_mm, color = 'b',linestyle='--')

						# PFD_FO
						plt.axhline(y = PFD_fo_lowerleft[1], xmin = PFD_fo_lowerleft[0]/vertical_mm, xmax = (PFD_fo_lowerleft[0] + PFD_fo_widthheight[0])/vertical_mm, color = 'w',linestyle='--')
						plt.axhline(y = PFD_fo_lowerleft[1]+ PFD_fo_widthheight[1], xmin = PFD_fo_lowerleft[0]/vertical_mm, xmax = (PFD_fo_lowerleft[0] + PFD_fo_widthheight[0])/vertical_mm, color = 'w',linestyle='--')
						plt.axvline(x = PFD_fo_lowerleft[0], ymin = PFD_fo_lowerleft[1]/horizontal_mm, ymax = (PFD_fo_lowerleft[1] + PFD_fo_widthheight[1])/horizontal_mm, color = 'w',linestyle='--')
						plt.axvline(x = PFD_fo_lowerleft[0]+ PFD_fo_widthheight[0], ymin = PFD_fo_lowerleft[1]/horizontal_mm, ymax = (PFD_fo_lowerleft[1] + PFD_fo_widthheight[1])/horizontal_mm, color = 'w',linestyle='--')

						# FIMD_FO
						plt.axhline(y = FIMD_fo_lowerleft[1], xmin = FIMD_fo_lowerleft[0]/vertical_mm, xmax = (FIMD_fo_lowerleft[0] + FIMD_fo_widthheight[0])/vertical_mm, color = 'r',linestyle='--')
						plt.axhline(y = FIMD_fo_lowerleft[1]+ FIMD_fo_widthheight[1], xmin = FIMD_fo_lowerleft[0]/vertical_mm, xmax = (FIMD_fo_lowerleft[0] + FIMD_fo_widthheight[0])/vertical_mm, color = 'r',linestyle='--')
						plt.axvline(x = FIMD_fo_lowerleft[0], ymin = FIMD_fo_lowerleft[1]/horizontal_mm, ymax = (FIMD_fo_lowerleft[1] + FIMD_fo_widthheight[1])/horizontal_mm, color = 'r',linestyle='--')
						plt.axvline(x = FIMD_fo_lowerleft[0]+ FIMD_fo_widthheight[0], ymin = FIMD_fo_lowerleft[1]/horizontal_mm, ymax = (FIMD_fo_lowerleft[1] + FIMD_fo_widthheight[1])/horizontal_mm, color = 'r',linestyle='--')

						# plt.show()
						# plt.savefig(crew_dir + "/Figures/" + 'smarteye_rightseat_'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0)
						# matplotlib.pyplot.close()
						###############

						## QAp Fig ##


						###############

			else:
					print("Empty: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
					pct_usable_matrix[i_scenario,i_seat] = np.nan
					
					if plot_heatmap_and_qatable:
						empty_heatmap = np.zeros((100,100))
						if file_types[i_seat] == "smarteye_leftseat":
							leftseat_heatmap[:,:,i_scenario] = empty_heatmap
						elif file_types[i_seat] == "smarteye_rightseat":
							rightseat_heatmap[:,:,i_scenario] = empty_heatmap

	if plot_heatmap_and_qatable:
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.style.use("dark_background")
		# plt.show()
		plt.savefig(path_to_project + "/Figures/" + 'smarteye_'+crews_to_process[i_crew]+'.tif',bbox_inches='tight',pad_inches=0)
		# Instead of plotting individual tables (embed percent text in subplot or in title of subplot?)
		matplotlib.pyplot.close()

	np.save(crew_dir + "/Processing/" + 'smarteye_pupild_leftseat',pupild_leftseat)
	np.save(crew_dir + "/Processing/" + 'smarteye_pupild_rightseat',pupild_rightseat)
	np.save(crew_dir + "/Processing/" + 'smarteye_headHeading_leftseat',headHeading_leftseat)
	np.save(crew_dir + "/Processing/" + 'smarteye_headHeading_rightseat',headHeading_rightseat)
	np.save(crew_dir + "/Processing/" + 'smarteye_timesec_epoch_storage',smarteye_timesec_epoch_storage)

	
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