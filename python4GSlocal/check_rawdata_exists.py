import pandas as pd
import os
import numpy as np
from os.path import exists
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from google.cloud import storage
import subprocess
import time

crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04', 'Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = ['Crew_01']
file_types = ["ifd_cockpit", "smarteye_leftseat","smarteye_rightseat", "abm_leftseat","abm_rightseat","emp_acc_leftseat","emp_acc_rightseat","emp_bvp_leftseat","emp_bvp_rightseat","emp_gsr_leftseat","emp_gsr_rightseat","emp_ibi_leftseat","emp_ibi_rightseat","emp_temp_leftseat","emp_temp_rightseat"]
scenarios = ["1","2","3","5","6","7","8","9"]


storage_client = storage.Client(project="soteria-fa59")
bucket = storage_client.bucket("soteria_study_data")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

###############################################

for this_crew in crews_to_process:
	file_existence_matrix = np.zeros((len(file_types),len(scenarios)))
	crew_dir = this_crew	
	print("checking files exists: " + crew_dir)
	# if not os.path.isdir(crew_dir + "/Figures"):
	# 	os.mkdir(crew_dir + "/Figures")
	for i_scenario in range(len(scenarios)):
		for i_devicefile in range(len(file_types)):
			process_dir_name = crew_dir + "/Processing/"
			blob = bucket.blob(process_dir_name + file_types[i_devicefile] + '_scenario' + scenarios[i_scenario] + '.csv')
			if blob.exists():
				print("file exists: " + process_dir_name + file_types[i_devicefile] + '_scenario' + scenarios[i_scenario] + '.csv')
				file_existence_matrix[i_devicefile,i_scenario] = 1
	

	if exists("Processing"):
		subprocess.Popen('rm -rf Processing', shell=True)
		time.sleep(5)
		os.mkdir("Processing")
	else:
		os.mkdir("Processing")

	if exists("Figures"):
		subprocess.Popen('rm -rf Figures', shell=True)
		time.sleep(5)
		os.mkdir("Figures")
	else:
		os.mkdir("Figures")

	fig, ax = plt.subplots()
	cbar_kws = { 'ticks' : [0, 1] }
	ax = sns.heatmap(file_existence_matrix, linewidths=.5, cbar_kws = cbar_kws)
	ax.set_xticklabels(scenarios)
	ax.set_yticklabels(file_types)
	ax.set(xlabel='scenarios', ylabel='devices')
	plt.yticks(rotation=0) 
	ax.xaxis.set_label_position('top') 
	ax.xaxis.tick_top()
	
	fig.tight_layout()
	#plt.show()
	plt.savefig('Figures/file_existence.jpg')
	matplotlib.pyplot.close()
	# os.remove(crew_dir + "/Processing/" + 'file_existence_matrix.npy')
	np.save("Processing/" + 'file_existence_matrix',file_existence_matrix)
	subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crew_dir + '"/Processing"', shell=True)
	subprocess.call('gsutil -m rsync -r Figures/ "gs://soteria_study_data/"'+ crew_dir + '"/Figures"', shell=True)