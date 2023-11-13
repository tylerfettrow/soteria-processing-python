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
import io
from google.cloud import storage
import shutil
crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']

storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

total_eventEEG_metric_dataframe = pd.DataFrame()
for i_crew in range(len(crews_to_process)):
	this_subject_event_metric_dataframe = pd.DataFrame()
	crew_dir = crews_to_process[i_crew]
	print("grabbing results from " + crew_dir)

	event_eegTimeSeries_metrics = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Processing/" +'event_eegTimeSeries_metrics' + '.csv'),delimiter=',')
	event_eegTimeSeries_df = event_eegTimeSeries_metrics[event_eegTimeSeries_metrics.columns[1:]]

	total_eventEEG_metric_dataframe = pd.concat([total_eventEEG_metric_dataframe,event_eegTimeSeries_df])

if exists("Analysis"):
	shutil.rmtree('Analysis', ignore_errors=True)
	time.sleep(5)
	os.mkdir("Analysis")
else:
	os.mkdir("Analysis")

total_eventEEG_metric_dataframe.to_csv("Analysis/" + 'total_eventEEG_metric_dataframe.csv')
subprocess.call('gsutil -m rsync -r Analysis/ "gs://soteria_study_data/"'+'"Analysis"', shell=True)

colors = sns.color_palette('tab10',2)
colors_array = np.array(colors)

total_eventEEG_metric_dataframe.seat[total_eventEEG_metric_dataframe.seat==1] = 'right'
total_eventEEG_metric_dataframe.seat[total_eventEEG_metric_dataframe.seat==0] = 'left'
total_eventEEG_metric_dataframe.event_label[total_eventEEG_metric_dataframe.event_label==0] = 'No Event'
total_eventEEG_metric_dataframe.event_label[total_eventEEG_metric_dataframe.event_label==1] = 'Event'
total_eventEEG_metric_dataframe.crew = total_eventEEG_metric_dataframe.crew.astype(int)

total_eventEEG_metric_dataframe.groupby(['crew','seat', 'event_label'])['taskLoad_index_spec'].mean().unstack(['seat', 'event_label']).plot.bar(color=[colors_array[0], colors_array[0]*.9, colors_array[1], colors_array[1]*.9],width=.9)
plt.title('taskLoad_index_spec', color='black')
plt.show()

total_eventEEG_metric_dataframe.groupby(['crew','seat', 'event_label'])['engagement_index_spec'].mean().unstack(['seat', 'event_label']).plot.bar(color=[colors_array[0], colors_array[0]*.9, colors_array[1], colors_array[1]*.9],width=.9)
plt.title('engagement_index_spec', color='black')
plt.show()