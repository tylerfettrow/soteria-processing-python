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

# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    ax.axis([pos, pos+10, ax.margins(y=.1), ax.margins(y=.1)])
    fig.canvas.draw_idle()
slider_color = 'White'				 

crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04', 'Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = ['Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = ['Crew_12']
file_types = ["abm_leftseat","abm_rightseat"]
# file_types = ["abm_leftseat"]
# sample_Rate = 260


scenarios = ["1","2","3","5","6","7"]
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'
plot_raw = 0
# plot_timeseries_psd = 1
number_of_epochs = 200

for i_crew in range(len(crews_to_process)):
	eeg_freq_storage = np.zeros((5,9,len(scenarios)))
	eeg_freq_band_storage = np.zeros((5,9,len(scenarios),number_of_epochs))
	eeg_timesec_epoch_storage = np.zeros((len(scenarios),number_of_epochs))
	crew_dir = path_to_project + '/' + crews_to_process[i_crew]
	for i_scenario in range(len(scenarios)):
		for i_seat in range(len(file_types)):
			process_dir_name = crew_dir + "/Processing/"
			if exists(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'):
				print("QA checking ECG: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				abm_data = pd.read_table((process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
				ch_names = ['F3','Fz','F4','C3','Cz','C4','P3','POz','P4']
				sfreq = 256
				info = mne.create_info(ch_names, sfreq,ch_types='eeg') # WARNING: for some reason 'eeg' multiplies amplitude by 1mill?? not sure why
				# data = {'UserTimeStamp':abm_data.UserTimeStamp, 'POz':abm_data.POz , 'Fz':abm_data.Fz, 'Cz':abm_data.Cz, 'C3':abm_data.C3, 'C4':abm_data.C4, 'F3':abm_data.F3, 'F4':abm_data.F4, 'P3':abm_data.P3, 'P4':abm_data.P4} 
				data = {'F3':abm_data.F3,'Fz':abm_data.Fz, 'F4':abm_data.F4, 'C3':abm_data.C3, 'Cz':abm_data.Cz, 'C4':abm_data.C4, 'P3':abm_data.P3,'POz':abm_data.POz , 'P4':abm_data.P4} 
				eeg = pd.DataFrame(data = data) #, abm_data.Fz, abm_data.Cz, abm_data.C3, abm_data.C4, abm_data.F3, abm_data.F4, abm_data.P3, abm_data.P4)
				eeg_trans = eeg.T/1000000 # adjusting for eeg multiplier
				raw = mne.io.RawArray(eeg_trans, info)
				
				# raw.plot(duration=100)

				# https://neuraldatascience.io/7-eeg/erp_filtering.html
				low_cut = 0.1
				hi_cut  = 50
				raw_filt = raw.copy().filter(low_cut, hi_cut)
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
				length_this_data = this_data.shape[1]

				for this_epoch in range(number_of_epochs):
					this_epoch_indices_start = np.floor(length_this_data/number_of_epochs) * this_epoch
					this_epoch_indices_end = this_epoch_indices_start + np.floor(length_this_data/number_of_epochs)
					eeg_timesec_epoch_storage[i_scenario,this_epoch] = abm_data.UserTimeStamp[this_epoch_indices_start]

					(f, S) = signal.welch(this_data[:,int(this_epoch_indices_start):int(this_epoch_indices_end)], sfreq)
					this_epoch_psd = 10 * np.log10(S.T) + 120 # convert to dB  # WARNING: no idea why 120 is needed here, but it results in similar values to the mne.compute_psd
					
					eeg_freq_band_storage[0,:,i_scenario,this_epoch] = this_epoch_psd[0:4,:].mean(0)   	# delta		
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
					np.save(crew_dir + "/Processing/" + 'eeg_freq_band_storage_leftseat', eeg_freq_band_storage)
					np.save(crew_dir + "/Processing/" + 'eeg_timesec_epoch_storage_leftseat', eeg_timesec_epoch_storage)
				if i_seat == 1:
					np.save(crew_dir + "/Processing/" + 'eeg_freq_band_storage_rightseat', eeg_freq_band_storage)
					np.save(crew_dir + "/Processing/" + 'eeg_timesec_epoch_storage_rightseat', eeg_timesec_epoch_storage)

				trial_psd_data = raw_filt.compute_psd()
				if plot_raw:
					trial_psd_data.plot()
				this_trial_data = trial_psd_data.get_data()
				this_trial_data = 10 * np.log10(this_trial_data) + 120 # convert to dB  # WARNING: no idea why 120 is needed here
				this_trial_data = this_trial_data.T


				# plt.plot(this_data)
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

				eeg_freq_storage[0,:,i_scenario] = this_trial_data[0:4,:].mean(0) # delta		
				eeg_freq_storage[1,:,i_scenario] = this_trial_data[4:8,:].mean(0) # theta
				eeg_freq_storage[2,:,i_scenario] = this_trial_data[8:13,:].mean(0) # alpha
				eeg_freq_storage[3,:,i_scenario] = this_trial_data[13:30,:].mean(0) # beta
				eeg_freq_storage[4,:,i_scenario] = this_trial_data[30:45,:].mean(0) # gamma

				if i_seat == 0:
					np.save(crew_dir + "/Processing/" + 'eeg_freq_storage_leftseat',eeg_freq_storage)
				if i_seat == 1:
					np.save(crew_dir + "/Processing/" + 'eeg_freq_storage_rightseat',eeg_freq_storage)


# #######################################


				# tstep = 10.0
				# events_ica = mne.make_fixed_length_events(raw_filt_grouped, duration=tstep)
				# epochs_ica = mne.Epochs(raw_filt_grouped, events_ica, tmin=0.0, tmax=tstep, baseline=None, preload=True)


				# fig = raw.compute_psd().plot()

				# random_state = 42   # ensures ICA is reproducable each time it's run
				# ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

				# Fit ICA
				# ica = mne.preprocessing.ICA(n_components=ica_n_components,
				#                             random_state=random_state)
				# ica.fit(epochs_ica, tstep=tstep)
				# ica.plot_components()
				# ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': hi_cut});


				# epochs_ica.plot_psd_topomap()
				# epochs_ica.plot_psd()
				# epochs_ica.average().plot_psd(fmax=50)

				# time = np.arange(data.size) / sfreq

				# fig, ax = plt.subplots(1, 1, figsize=(12, 4))
				# plt.plot(eeg.Cz, lw=1.5, color='k')
				# plt.xlabel('Time (seconds)')
				# plt.ylabel('Voltage')
				# # plt.xlim([time.min(), time.max()])
				# plt.title('N3 sleep EEG data (F3)')
				# sns.despine()

				# # https://raphaelvallat.com/bandpower.html
				# win = 60 * sfreq
				# freqs, psd = signal.welch(eeg.Cz, sfreq, nperseg=win)
				# sns.set(font_scale=1.2, style='white')
				# plt.figure(figsize=(8, 4))
				# plt.plot(freqs, psd, color='k', lw=2)
				# sns.despine()
				# plt.show()

				# sns.set(style="white", font_scale=1.2)
			 #    # Compute the PSD
			 #    # freqs, psd = periodogram(eeg.Cz, sf)
			 #    # freqs_welch, psd_welch = welch(eeg.Cz, sf, nperseg=window_sec*sf)
			 #    psd_mt, freqs_mt = psd_array_multitaper(eeg.Cz, sfreq, adaptive=True, normalization='full', verbose=0)
			 #    sharey = False

			 #    # Optional: convert power to decibels (dB = 10 * log10(power))
			 #    psd = 10 * np.log10(psd)
		  #       psd_welch = 10 * np.log10(psd_welch)
		  #       psd_mt = 10 * np.log10(psd_mt)
		  #       sharey = True

			 #    # Start plot
			 #    fig, ax = plt.subplots(1, 1, figsize=(12, 4), sharex=True, sharey=sharey)
			 #    # Stem
			 #    sc = 'slategrey'
			 #    # ax1.stem(freqs, psd, linefmt=sc, basefmt=" ", markerfmt=" ")
			 #    # ax2.stem(freqs_welch, psd_welch, linefmt=sc, basefmt=" ", markerfmt=" ")
			 #    ax.stem(freqs_mt, psd_mt, linefmt=sc, basefmt=" ", markerfmt=" ")
			 #    # Line
			 #    lc, lw = 'k', 2
			 #    # ax1.plot(freqs, psd, lw=lw, color=lc)
			 #    # ax2.plot(freqs_welch, psd_welch, lw=lw, color=lc)
			 #    ax.plot(freqs_mt, psd_mt, lw=lw, color=lc)

				# ecg = pd.DataFrame(abm_data.UserTimeStamp, abm_data.ECG)
				# ecg = abm_data[['UserTimeStamp','ECG']].set_index('UserTimeStamp')
				# peaks, _ = find_peaks(abm_data.ECG, distance=100,prominence=500, width=1)

				# fig, ax = plt.subplots()
				# plt.plot(abm_data.UserTimeStamp, abm_data.ECG)
				# plt.plot(abm_data.UserTimeStamp[peaks],abm_data.ECG[peaks],"x")
				# # ecg.plot(ax=ax)
				# plt.title('RawData: ' + file_types[i_seat] + '_scenario' + scenarios[i_scenario])
				# axis_position = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor = slider_color)
				# slider_position = Slider(axis_position, 'Pos', 0.1, 90.0)				 
				# slider_position.on_changed(update) 
				# plt.show()