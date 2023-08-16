import pandas as pd
import numpy as np
import os
from os.path import exists
import mne
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from mne.time_frequency import psd_array_multitaper
from scipy.signal import welch, periodogram
from google.cloud import storage
from numpy import linalg as la
from os.path import exists
import subprocess
import time
from tensorflow.python.lib.io import file_io
import io
from PIL import Image

crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_02']
# file_types = ["leftseat","rightseat"]
# scenarios = ["1","2","3","5","6","7"]
# fig_types = ["engagement", "taskLoad"]
fig_types = ["engagement_spec"]
fig_types = ["taskLoad_spec"]
storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

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

for i_crew in range(len(crews_to_process)):

	crew_dir = crews_to_process[i_crew]
	figure_dir_name = crew_dir + '/Figures/'
	this_file_name = 'eeg_' + fig_types[0] + "_" + crew_dir + '.tif'
	print('eeg_'+fig_types[0]+crew_dir+'.tif')

	blob = bucket.blob(figure_dir_name + this_file_name)
	
	blob.download_to_filename('Figures/' + this_file_name)

	# im = Image.open('gs://soteria_study_data/' + figure_dir_name + this_file_name)
	# im.show()

	input("Press Enter to continue...")


	# subprocess.call('gsutil -m rsync -r Figures/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Figures"', shell=True)


# if engagement
# fig_types = ["engagement"]
# for each crew, grab and display this tiff
# create function that waits for button press, key left moves for loop back 1, and key right move for loop up 1
