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

storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

total_eventSmarteye_metric_dataframe = pd.DataFrame()
for i_crew in range(len(crews_to_process)):
	this_subject_event_metric_dataframe = pd.DataFrame()
	crew_dir = crews_to_process[i_crew]
	print("grabbing results from " + crew_dir)

	event_smarteyeGazeTimeSeries_metrics = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Processing/" +'event_smarteyeGazeTimeSeries_metrics' + '.csv'),delimiter=',')
	event_smarteyeGazeTimeSeries_df = event_smarteyeGazeTimeSeries_metrics[event_smarteyeGazeTimeSeries_metrics.columns[1:]]

	event_smarteyeTimeSeries_metrics = pd.read_table(('gs://soteria_study_data/' + crew_dir + "/Processing/" +'event_smarteyeTimeSeries_metrics' + '.csv'),delimiter=',')
	event_smarteyeTimeSeries_df = event_smarteyeTimeSeries_metrics[event_smarteyeTimeSeries_metrics.columns[6:]]

	this_subject_event_metric_dataframe = event_smarteyeGazeTimeSeries_df.join(event_smarteyeTimeSeries_df)

	total_eventSmarteye_metric_dataframe = pd.concat([total_eventSmarteye_metric_dataframe,this_subject_event_metric_dataframe])

if exists("Analysis"):
	subprocess.Popen('rm -rf Analysis', shell=True)
	time.sleep(5)
	os.mkdir("Analysis")
else:
	os.mkdir("Analysis")

total_eventSmarteye_metric_dataframe.to_csv("Analysis/" + 'total_eventSmarteye_metric_dataframe.csv')
subprocess.call('gsutil -m rsync -r Analysis/ "gs://soteria_study_data/"'+'"Analysis"', shell=True)

colors = sns.color_palette('tab10',2)
colors_array = np.array(colors)

total_eventSmarteye_metric_dataframe.seat[total_eventSmarteye_metric_dataframe.seat==1] = 'right'
total_eventSmarteye_metric_dataframe.seat[total_eventSmarteye_metric_dataframe.seat==0] = 'left'
total_eventSmarteye_metric_dataframe.event_label[total_eventSmarteye_metric_dataframe.event_label==0] = 'No Event'
total_eventSmarteye_metric_dataframe.event_label[total_eventSmarteye_metric_dataframe.event_label==1] = 'Event'
total_eventSmarteye_metric_dataframe.crew = total_eventSmarteye_metric_dataframe.crew.astype(int)

total_eventSmarteye_metric_dataframe.groupby(['crew','seat', 'event_label'])['gaze_variance'].mean().unstack(['seat', 'event_label']).plot.bar(color=[colors_array[0], colors_array[0]*.9, colors_array[1], colors_array[1]*.9],width=.9)
plt.title('gaze_variance', color='black')
plt.show()

total_eventSmarteye_metric_dataframe.groupby(['crew','seat', 'event_label'])['gaze_vel_avg'].mean().unstack(['seat', 'event_label']).plot.bar(color=[colors_array[0], colors_array[0]*.9, colors_array[1], colors_array[1]*.9],width=.9)
plt.title('gaze_vel_avg', color='black')
plt.show()

total_eventSmarteye_metric_dataframe.groupby(['crew','seat', 'event_label'])['gaze_vel_std'].mean().unstack(['seat', 'event_label']).plot.bar(color=[colors_array[0], colors_array[0]*.9, colors_array[1], colors_array[1]*.9],width=.9)
plt.title('gaze_vel_std', color='black')
plt.show()

total_eventSmarteye_metric_dataframe.groupby(['crew','seat', 'event_label'])['headHeading_avg'].mean().unstack(['seat', 'event_label']).plot.bar(color=[colors_array[0], colors_array[0]*.9, colors_array[1], colors_array[1]*.9],width=.9)
plt.title('headHeading_avg', color='black')
plt.show()

total_eventSmarteye_metric_dataframe.groupby(['crew','seat', 'event_label'])['headHeading_std'].mean().unstack(['seat', 'event_label']).plot.bar(color=[colors_array[0], colors_array[0]*.9, colors_array[1], colors_array[1]*.9],width=.9)
plt.title('headHeading_std', color='black')
plt.show()

total_eventSmarteye_metric_dataframe.groupby(['crew','seat', 'event_label'])['pupilD_avg'].mean().unstack(['seat', 'event_label']).plot.bar(color=[colors_array[0], colors_array[0]*.9, colors_array[1], colors_array[1]*.9],width=.9)
plt.title('pupilD_avg', color='black')
plt.show()

total_eventSmarteye_metric_dataframe.groupby(['crew','seat', 'event_label'])['pupilD_std'].mean().unstack(['seat', 'event_label']).plot.bar(color=[colors_array[0], colors_array[0]*.9, colors_array[1], colors_array[1]*.9],width=.9)
plt.title('pupilD_std', color='black')
plt.show()