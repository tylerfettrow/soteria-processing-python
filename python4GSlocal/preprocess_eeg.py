import pandas as pd
import numpy as np
import os
from os.path import exists
import mne
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import signal
from mne.time_frequency import psd_array_multitaper
from scipy.signal import welch, periodogram
from google.cloud import storage
from numpy import linalg as la
from os.path import exists
import subprocess
import time
import matplotlib.pyplot as plt
import pywt

# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    ax.axis([pos, pos+10, ax.margins(y=.1), ax.margins(y=.1)])
    fig.canvas.draw_idle()
slider_color = 'White'				 


crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_01']
file_types = ["abm_leftseat","abm_rightseat"]
scenarios = ["1","2","3","5","6","7"]
storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

scenarios = ["1","2","3","5","6","7"]
# path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'
plot_raw = 0
# plot_timeseries_psd = 1
number_of_epochs = 1000

for i_crew in range(len(crews_to_process)):
	eeg_freq_storage = np.zeros((5,9,len(scenarios)))
	eeg_freq_band_storage = np.zeros((5,9,len(scenarios),number_of_epochs))
	eeg_freqSpec_band_storage = np.zeros((6,9,2000,len(scenarios)))
	eeg_freqSpec_band_storage[:] = np.nan
	eeg_timesec_epoch_storage = np.zeros((len(scenarios),number_of_epochs))
	crew_dir = crews_to_process[i_crew]
	
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

	for i_seat in range(len(file_types)):
		for i_scenario in range(len(scenarios)):	
			# eeg_freq_band_storage = None
			# eeg_freq_band_storage = np.zeros((4,9,len(scenarios),number_of_epochs))
			process_dir_name = crew_dir + "/Processing/"
			blob = bucket.blob(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
			if blob.exists():
				# smarteye_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
				abm_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
				
				time_end = abm_data.UserTimeStamp[abm_data.UserTimeStamp.shape[0]-1]

				print("QA checking ECG: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				
				ch_names = ['F3','Fz','F4','C3','Cz','C4','P3','POz','P4']
				sfreq = 256
				info = mne.create_info(ch_names, sfreq,ch_types='eeg') # WARNING: for some reason 'eeg' multiplies amplitude by 1mill?? not sure why
				# data = {'UserTimeStamp':abm_data.UserTimeStamp, 'POz':abm_data.POz , 'Fz':abm_data.Fz, 'Cz':abm_data.Cz, 'C3':abm_data.C3, 'C4':abm_data.C4, 'F3':abm_data.F3, 'F4':abm_data.F4, 'P3':abm_data.P3, 'P4':abm_data.P4} 
				data = {'F3':abm_data.F3,'Fz':abm_data.Fz, 'F4':abm_data.F4, 'C3':abm_data.C3, 'Cz':abm_data.Cz, 'C4':abm_data.C4, 'P3':abm_data.P3,'POz':abm_data.POz , 'P4':abm_data.P4} 
				eeg = pd.DataFrame(data = data) #, abm_data.Fz, abm_data.Cz, abm_data.C3, abm_data.C4, abm_data.F3, abm_data.F4, abm_data.P3, abm_data.P4)
				eeg_trans = eeg.T/1000000 # adjusting for eeg multiplier
				raw = mne.io.RawArray(eeg_trans, info)

				# https://neuraldatascience.io/7-eeg/erp_filtering.html
				low_cut = 0.1
				hi_cut  = 50
				raw_data = raw.copy()
				raw_filt = raw.copy().filter(low_cut, hi_cut)
				raw_filt
				# raw_filt.plot_psd(fmax=100)
				if plot_raw:
					raw_filt.plot(duration=100,block=True)

				# Number	labels	theta	radius	X	Y	Z	sph_theta	sph_phi	sph_radius	type	
				# 0 # 1	Fz	-89.7	0.23	0.312	58.5	66.5	89.7	48.6	88.5	   	
				# 1 # 2	F3	-133	0.333	-50.2	53.1	42.2	133	 30	84.4	   	
				# 2 # 3	F4	-46.3	0.341	51.8	54.3	40.8	46.3	28.5	85.5	 
				# 3 # 4	Cz	87.5	0.0291	0.401	-9.17	100	-87.5	84.8	101	   	
				# 4 # 5	C3	170	0.255	-65.4	-11.6	64.4	-170	44.1	92.5	   	
				# 5 # 6	C4	9.22	0.261	67.1	-10.9	63.6	-9.22	43.1	93.1	   	
				# 6 # 7	POz	89.9	0.354	0.216	-102	50.6	-89.9	26.3	114	   	  	
				# 7 # 8	P3	124	0.331	-53	-78.8	55.9	-124	30.5	110	   	
				# 8 # 9	P4	54.7	0.331	55.7	-78.6	56.6	-54.7	30.4	112	   


				# 				#####################	
				# easycap_montage = mne.channels.make_standard_montage('easycap-M1')
				# raw_filt.set_montage('easycap-M1')
				# fig = raw_filt.plot_sensors(show_names=True)

				# if crews_to_process[i_crew] == 'Crew_02' & scenarios[i_scenario] == "6":			
				# 	groups=dict(Front=[1, 5, 6], Middle=[2, 3, 4], Posterior=[0, 7, 8])
				# 	raw_filt_grouped = mne.channels.combine_channels(raw_filt, groups, method='mean', drop_bad=True)
				# else:
				# groups=dict(Front=[1, 5, 6], Middle=[2, 3, 4], Posterior=[0, 7, 8])
				# raw_filt_grouped = mne.channels.combine_channels(raw_filt, groups, method='mean', drop_bad=True)

				# this_data = raw_filt_grouped.get_data()
				

				this_data = raw_filt.get_data()
				this_data = this_data + .0005
				length_this_data = this_data.shape[1]

				for this_epoch in range(number_of_epochs):
					this_epoch_indices_start = np.floor(length_this_data/number_of_epochs) * this_epoch
					this_epoch_indices_end = this_epoch_indices_start + np.floor(length_this_data/number_of_epochs)
					eeg_timesec_epoch_storage[i_scenario,this_epoch] = abm_data.UserTimeStamp[this_epoch_indices_start]

					# f, Pxx_spec = signal.periodogram(this_data[:,int(this_epoch_indices_start):int(this_epoch_indices_end)], 256, 'flattop', scaling='spectrum')
					# this_epoch_psd = 10 * np.log10(Pxx_spec.T)
					(f, S) = signal.welch(this_data[:,int(this_epoch_indices_start):int(this_epoch_indices_end)],nperseg= 2048, fs = sfreq, scaling = 'density')
					this_epoch_psd = (S.T)
					this_epoch_psd = 10 * np.log10(S.T) + 120 # convert to dB  # WARNING: no idea why 120 is needed here, but it results in similar values to the mne.compute_psd
					
					eeg_freq_band_storage[0,:,i_scenario,this_epoch] = this_epoch_psd[1:4,:].mean(0)   	# delta		
					eeg_freq_band_storage[1,:,i_scenario,this_epoch] = this_epoch_psd[4:8,:].mean(0)   	# theta
					eeg_freq_band_storage[2,:,i_scenario,this_epoch] = this_epoch_psd[8:13,:].mean(0)  	# alpha
					eeg_freq_band_storage[3,:,i_scenario,this_epoch] = this_epoch_psd[13:30,:].mean(0) 	# beta
					eeg_freq_band_storage[4,:,i_scenario,this_epoch] = this_epoch_psd[30:45,:].mean(0) 	# gamma
				
				# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4712412/
				# if plot_timeseries_psd:
				# plt.plot(eeg_freq_band_storage[0,1,0,:])
				# plt.plot(eeg_freq_band_storage[1,1,0,:])
				# plt.plot(eeg_freq_band_storage[2,1,0,:])
				# plt.plot(eeg_freq_band_storage[3,1,0,:])
				# plt.plot(eeg_freq_band_storage[4,1,0,:])
				# plt.legend(['delta', 'theta', 'alpha', 'beta', 'gamma'])
				
				if i_seat == 0:
					np.save("Processing/" + 'eeg_freq_band_storage_leftseat', eeg_freq_band_storage)
					np.save("Processing/" + 'eeg_timesec_epoch_storage_leftseat', eeg_timesec_epoch_storage)
				if i_seat == 1:
					np.save("Processing/" + 'eeg_freq_band_storage_rightseat', eeg_freq_band_storage)
					np.save("Processing/" + 'eeg_timesec_epoch_storage_rightseat', eeg_timesec_epoch_storage)

				trial_psd_data = raw_filt.compute_psd()
				# if plot_raw:
				# 	trial_psd_data.plot()
				this_trial_data = trial_psd_data.get_data()
				this_trial_data = 10 * np.log10(this_trial_data) + 120# convert to dB  # WARNING: no idea why 120 is needed here
				this_trial_data = this_trial_data.T
				freqs = np.linspace(0,128,1025)
				# plt.plot(freqs,this_trial_data)
				# plt.show()


				# find the psd of grouped data
				# for each group (column)
				# grab the average power (dB) for (vv indices vv) 
				# 	1-3 (delta)
				#   4-7 (theta)
				#   8-12 (alpha)
				#   13-30 (beta)
				#   30-45 (gamma)

				# store in a matrix (5 x 3 x seat)
				# if group has data, do this, otherwise, store NA
				# eeg_freq_storage[0,0,i_seat] = # frontal delta

				eeg_freq_storage[0,:,i_scenario] = this_trial_data[freqs<4,:].mean(0) # delta
				eeg_freq_storage[1,:,i_scenario] = this_trial_data[(freqs>=4)&(freqs<8),:].mean(0) # theta
				eeg_freq_storage[2,:,i_scenario] = this_trial_data[(freqs>=8)&(freqs<13),:].mean(0) # alpha
				eeg_freq_storage[3,:,i_scenario] = this_trial_data[(freqs>=13)&(freqs<30),:].mean(0) # beta
				eeg_freq_storage[4,:,i_scenario] = this_trial_data[(freqs>=30)&(freqs<45),:].mean(0) # gamma

				if i_seat == 0:
					np.save("Processing/" + 'eeg_freq_storage_leftseat',eeg_freq_storage)
				if i_seat == 1:
					np.save("Processing/" + 'eeg_freq_storage_rightseat',eeg_freq_storage)


				waveletname = 'morl'
				cmap = 'seismic'
				dt = 1/sfreq
				this_data_dwt = this_data.T
				N = this_data_dwt.shape[0]
				time = np.linspace(0, time_end, N)
				
				# scales = np.arange(1, 60)
				# plot_wavelet(time, signal, scales)
				# [coefficients, frequencies] = pywt.cwt(this_data_dwt[:,0].T, scales,'gaus1')

				# plt.figure(figsize=(15, 10))
				# plt.imshow(abs(coefficients), extent=[0, 200, 30, 1], interpolation='bilinear', cmap='bone',
				           # aspect='auto', vmax=abs(coefficients).max(), vmin=abs(coefficients).min())
				# plt.gca().invert_yaxis()
				# plt.yticks(np.arange(1, 31, 1))
				# plt.xticks(np.arange(0, 201, 10))
				# plt.show()


				# [coefficients, frequencies] = pywt.cwt(this_data_dwt[:,0], scales, waveletname, dt)
				# power = np.power((abs(coefficients)) ** 2,-1)
				# period = 1. / frequencies
			    
				# fig, axs = plt.subplots(2)
				# # fig, axs[0] = plt.subplots(figsize=(15, 10))
				# axs[0].plot(time, this_data_dwt[:,0])
				# axs[0].set_xlim(0, this_data_dwt.shape[0])
				# im = axs[1].contourf(time, frequencies, power, extend='both',cmap=cmap)
				# # im = axs[1].contourf(time, (period), (power))
				# # ax.get_legend().remove()
				# # ax.set_title(title, fontsize=20)
				# axs[1].set_ylabel("Period (?)", fontsize=18)
				# axs[1].set_xlabel("time (seconds)", fontsize=18)
				# axs[1].invert_yaxis()
				# # fig.colorbar(im, orientation="vertical")
				# plt.show()


				# plt.specgram(this_data_dwt[:,0].T, NFFT=1024, Fs=44100, noverlap=900)
				# fig.colorbar(orientation="vertical")

				# fig, ax = plt.subplots(2)
				# ax[0].plot(time, this_data_dwt[:,0])
				# ax[0].set_xlim(0, time[-1])
				f, t, Sxx = signal.spectrogram(this_data_dwt[:,:].T, 256)
				# print(this_data_dwt[:,:].T.shape)
				# print(np.floor(this_data_dwt[:,:].T.shape[1]/1000))
				# print(Sxx.shape)
			
				fmin = 1 # Hz
				fmax = 45 # Hz
				freq_slice = np.where((f >= fmin) & (f <= fmax))

				# keep only frequencies of interest
				f   = f[freq_slice]
				Sxx = np.squeeze(Sxx[:,freq_slice,:]) * 10000000000000

				# im = ax[1].pcolormesh(t[600:620], np.fft.fftshift(f), np.fft.fftshift(Sxx[:,600:620]))
				# # fig.colorbar(im, orientation="vertical")
				# ax[1].set_ylabel('Frequency [Hz]')
				# ax[1].set_xlabel('Time [sec]')
				# plt.show()
				
				# for this_epoch in range(Sxx.shape[2]):
				# 	this_epoch_indices_start = np.floor(length_this_data/number_of_epochs) * this_epoch
				# 	this_epoch_indices_end = this_epoch_indices_start + np.floor(length_this_data/number_of_epochs)
				# 	eeg_timesec_epoch_storage[i_scenario,this_epoch] = abm_data.UserTimeStamp[this_epoch_indices_start]
				eeg_freqSpec_band_storage[0,:,0:Sxx.shape[2],i_scenario] = t
				eeg_freqSpec_band_storage[1,:,0:Sxx.shape[2],i_scenario] = Sxx[:,0:3, :].mean(1)   	# delta
				eeg_freqSpec_band_storage[2,:,0:Sxx.shape[2],i_scenario] = Sxx[:,4:7, :].mean(1)		# theta		
				eeg_freqSpec_band_storage[3,:,0:Sxx.shape[2],i_scenario] = Sxx[:,8:12, :].mean(1)		# alpha
				eeg_freqSpec_band_storage[4,:,0:Sxx.shape[2],i_scenario] = Sxx[:,13:29, :].mean(1)		# beta
				eeg_freqSpec_band_storage[5,:,0:Sxx.shape[2],i_scenario] = Sxx[:,30:44, :].mean(1)		# gamma
				# print(f)

				if i_seat == 0:
					np.save("Processing/" + 'eeg_freqSpec_band_storage_leftseat', eeg_freqSpec_band_storage)
					# np.save("Processing/" + 'eeg_timesec_epoch_storage_leftseat', eeg_timesec_epoch_storage)
				if i_seat == 1:
					np.save("Processing/" + 'eeg_freqSpec_band_storage_rightseat', eeg_freqSpec_band_storage)
					# np.save("Processing/" + 'eeg_timesec_epoch_storage_rightseat', eeg_timesec_epoch_storage)


				# 	eeg_freqSpec_band_storage[1,:,i_scenario] = Sxx[:,1:4, :].mean(1)   	# delta
				# 	eeg_freqSpec_band_storage[2,:,i_scenario] = Sxx[:,4:8, :].mean(1)		# theta		
				# 	eeg_freqSpec_band_storage[3,:,i_scenario] = Sxx[:,8:13, :].mean(1)		# alpha
				# 	eeg_freqSpec_band_storage[4,:,i_scenario] = Sxx[:,13:30, :].mean(1)		# beta
				# 	eeg_freqSpec_band_storage[5,:,i_scenario] = Sxx[:,30:45, :].mean(1)		# gamma

	subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Processing"', shell=True)