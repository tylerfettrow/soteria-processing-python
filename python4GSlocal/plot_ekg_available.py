import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import helpers
import importlib
import seaborn as sns; sns.set_theme()

importlib.reload(helpers)

helper = helpers.HELP()
helper.reset_folder_storage()


crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03']
# crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04','Crew_05', 'Crew_06']
scenarios = ["1","2","3","5","6","7"]

total_available_matrix = np.zeros((len(scenarios),2,len(crews_to_process)))

for i_crew in range(len(crews_to_process)):
	crew_dir = crews_to_process[i_crew]
	print(crew_dir)
	this_matrix = pd.read_table(('gs://soteria_study_data/'+ crew_dir + '/Processing/ekg_pct_usable_matrix.csv'),delimiter=',')
	this_matrix = np.array(this_matrix)
	total_available_matrix[:,:,i_crew] = this_matrix[:,1:]

total_data_available_matrix_reshaped = total_available_matrix.transpose(0,2,1).reshape(6,total_available_matrix.shape[1]*total_available_matrix.shape[2])

# fig, ax = plt.subplots()
# cbar_kws = { 'ticks' : [0, 100] }
# ax = sns.heatmap(np.rint(total_data_available_matrix_reshaped), linewidths=.5, cbar_kws = cbar_kws,annot=True,fmt='.3g')
# # ax.set_xticklabels(scenarios)
# # ax.set_yticklabels(file_types)
# # ax.set(xlabel='scenarios', ylabel='devices')
# # plt.yticks(rotation=0)
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()

# # fig.tight_layout()
# # plt.show()
# plt.savefig("Figures/" + 'total_ekg_available_matrix.jpg')
# # matplotlib.pyplot.close()

fig, ax = plt.subplots()
cbar_kws = { 'ticks' : [0, 100] }
ax = sns.heatmap(np.rint(total_data_available_matrix_reshaped), linewidths=.5,cbar = False, cbar_kws = cbar_kws,annot=True,fmt='.3g')
# ax.set_xticklabels(scenarios)
# ax.set_yticklabels(file_types)
# ax.set(xlabel='scenarios', ylabel='devices')
plt.axis('off')
plt.yticks(rotation=0)
# ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

fig.tight_layout()
# plt.show()
plt.savefig("Figures/" + 'total_ekg_available_matrix.jpg')
# matplotlib.pyplot.close()

