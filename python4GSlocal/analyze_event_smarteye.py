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
from sklearn import linear_model
import statsmodels.api as sm

crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
file_types = ["smarteye_leftseat","smarteye_rightseat"]
scenarios = ["1","2","3","5","6","7","8","9"]

storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")


for i_crew in range(len(crews_to_process)):
	pct_usable_matrix = np.zeros((len(scenarios),len(file_types)))
	crew_dir = crews_to_process[i_crew]
	process_dir_name = crew_dir + "/Processing/"

	f_stream = file_io.FileIO('gs://soteria_study_data/'+ crew_dir + "/Processing/" + 'event_vector_scenario.npy', 'rb')
	this_event_data = np.load(io.BytesIO(f_stream.read()))

	for i_scenario in range(len(scenarios)):
		for i_seat in range(len(file_types)):
			blob = bucket.blob(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
			if blob.exists():
				print("QA checking: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				smarteye_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')

				rpsa_df = pd.read_excel('gs://soteria_study_data/rpsa.xlsx',header=0)