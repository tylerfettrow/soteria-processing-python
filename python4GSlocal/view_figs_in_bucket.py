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


crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_02']
file_types = ["leftseat","rightseat"]
scenarios = ["1","2","3","5","6","7"]

storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")



for i_crew in range(len(crews_to_process)):

	crew_dir = crews_to_process[i_crew]
	process_dir_name = crew_dir + '/Processing/'

	f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.npy', 'rb')
	this_event_data = np.load(io.BytesIO(f_stream.read()))