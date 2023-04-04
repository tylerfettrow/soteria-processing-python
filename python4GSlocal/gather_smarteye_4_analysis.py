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
from tensorflow.python.lib.io import file_io
import io
from google.cloud import storage

crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_01','Crew_02'];
file_types = ["smarteye_leftseat","smarteye_rightseat"]
scenarios = ["1","2","3","5","6","7"]

# event_smarteyeGaze_metrics = np.zeros((len(scenarios)*2,12))
# event_smarteyeGaze_metrics[:, 0] = int(i_crew + 1)
event_smarteyeGaze_column_values = ['crew', 'seat', 'scenario', 'gaze_variance_control', 'gaze_variance_event1', 'gaze_variance_event2', 'gaze_vel_avg_control', 'gaze_vel_avg_event1', 'gaze_vel_avg_event2', 'gaze_vel_std_control', 'gaze_vel_std_event1', 'gaze_vel_std_event2']

# event_smarteyeTime_metrics = np.zeros((len(scenarios)*2,15))
# event_smarteyeTime_metrics[:, 0] = int(i_crew + 1)
event_smarteyeTime_column_values = ['crew', 'seat', 'scenario', 'headHeading_avg_control', 'headHeading_avg_event1', 'headHeading_avg_event2', 'headHeading_std_control', 'headHeading_std_event1', 'headHeading_std_event2', 'pupilD_avg_control', 'pupilD_avg_event1', 'pupilD_avg_event2', 'pupilD_std_control', 'pupilD_std_event1', 'pupilD_std_event2']


storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

total_eventSmarteye_metric_dataframe = pd.DataFrame()
for i_crew in range(len(crews_to_process)):
	crew_dir = crews_to_process[i_crew]
	print("grabbing results from " + crew_dir)
	# process_dir_name = crew_dir + "/Processing/"

	f_stream = file_io.FileIO('gs://soteria_study_data/'+ crew_dir + "/Processing/" + 'event_smarteyeGaze_metrics.npy', 'rb')
	this_eventGaze_metric_data = np.load(io.BytesIO(f_stream.read()))
	this_eventGaze_metric_dataframe = pd.DataFrame(this_eventGaze_metric_data, columns = event_smarteyeGaze_column_values)


	f_stream = file_io.FileIO('gs://soteria_study_data/'+ crew_dir + "/Processing/" + 'event_smarteyeTime_metrics.npy', 'rb')
	this_eventTime_metric_data = np.load(io.BytesIO(f_stream.read()))
	this_eventTime_metric_dataframe = pd.DataFrame(this_eventTime_metric_data[:,3:], columns = event_smarteyeTime_column_values[3:])

	this_subject_event_metric_dataframe = this_eventGaze_metric_dataframe.join(this_eventTime_metric_dataframe)

	total_eventSmarteye_metric_dataframe = pd.concat([total_eventSmarteye_metric_dataframe,this_subject_event_metric_dataframe])

if exists("Analysis"):
	subprocess.Popen('rm -rf Analysis', shell=True)
	time.sleep(5)
	os.mkdir("Analysis")
else:
	os.mkdir("Analysis")

total_eventSmarteye_metric_dataframe.to_csv("Analysis/" + 'total_eventSmarteye_metric_dataframe.csv')
subprocess.call('gsutil -m rsync -r Analysis/ "gs://soteria_study_data/"'+'"Analysis"', shell=True)


