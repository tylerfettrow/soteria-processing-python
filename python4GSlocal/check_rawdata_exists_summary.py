import pandas as pd
import os
import numpy as np
from os.path import exists
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import io
import numpy as np
from tensorflow.python.lib.io import file_io
import subprocess
import time


crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04', 'Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_01']
file_types = ["ifd_cockpit", "smarteye_leftseat","smarteye_rightseat", "abm_leftseat","abm_rightseat","emp_acc_leftseat","emp_acc_rightseat","emp_bvp_leftseat","emp_bvp_rightseat","emp_gsr_leftseat","emp_gsr_rightseat","emp_ibi_leftseat","emp_ibi_rightseat","emp_temp_leftseat","emp_temp_rightseat"];
scenarios = ["1","2","3","5","6","7","8","9"];

# path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'

total_file_existence_matrix = np.zeros((len(file_types),len(scenarios),len(crews_to_process)))
#print(total_file_existence_matrix.shape)
for i_crew in range(len(crews_to_process)):
	crew_dir = crews_to_process[i_crew]
	f_stream = file_io.FileIO('gs://soteria_study_data/'+ crew_dir + '/Processing/file_existence_matrix.npy', 'rb')
	this_matrix = np.load(io.BytesIO(f_stream.read()))
	total_file_existence_matrix[:,:,i_crew] = this_matrix
	

#print(total_file_existence_matrix[:,:,0])
total_file_existence_matrix_squeezed = np.squeeze(np.sum(total_file_existence_matrix,axis=2))
total_file_existence_matrix_squeezed_averaged = np.multiply(np.divide(total_file_existence_matrix_squeezed,len(crews_to_process)), 100)

#print(np.multiply(np.divide(np.squeeze(np.sum(total_file_existence_matrix,axis=2)),len(crews_to_process)), 100))

#total_file_existence_matrix_percent = squeeze(sum(total_file_existence_matrix,3))./ size(total_file_existence_matrix,3) .* 100; %percent


if exists("Figures"):
	subprocess.Popen('rm -rf Figures', shell=True)
	time.sleep(5)
	os.mkdir("Figures")
else:
	os.mkdir("Figures")


fig, ax = plt.subplots()
cbar_kws = { 'ticks' : [0, 100] }
ax = sns.heatmap(np.rint(total_file_existence_matrix_squeezed_averaged), linewidths=.5, cbar_kws = cbar_kws,annot=True,fmt='.3g')
ax.set_xticklabels(scenarios)
ax.set_yticklabels(file_types)
ax.set(xlabel='scenarios', ylabel='devices')
plt.yticks(rotation=0)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

fig.tight_layout()
#plt.show()
plt.savefig("Figures/" + 'total_file_existence.jpg')
matplotlib.pyplot.close()
subprocess.call('gsutil -m rsync -r Figures/ "gs://soteria_study_data/Figures"', shell=True)