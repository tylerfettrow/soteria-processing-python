% Pyshio QA
% expected to be run within study folder ("SOTERIA")

% DATA PRESENT
% for each file within each trial
% 1) determine whether there is data present for every emp folder
% 2) generate heat map of whether data present (green good red bad)
% % % we know what there "should be", so ok to hard code that
% 3) save the heatmap fig in "figures"

clear,clc
crew_ids = {'Crew1'};
% crew_ids = {'Crew2','Crew3','Crew4','Crew5','Crew6'};
% file_types = {'abm_leftseat','abm_rightseat'};
% scenarios = {'1','2','3','5','6','7','8','9'};
data_dir = pwd;
% set(groot, 'defaultAxesTickLabelInterpreter','none'); set(groot, 'defaultLegendInterpreter','none');

for this_crew = 1:length(crew_ids) 
    fileID = fopen(fullfile(crew_ids{this_crew},'trial_settings.txt'), 'r');
    matlab_path = fullfile(crew_ids{this_crew},'matlab');
    
    text_line = fgetl(fileID);
    text_cell = {};
    while ischar(text_line)
        text_cell = [text_cell; text_line]; %#ok<AGROW>
        text_line = fgetl(fileID);
    end
    fclose(fileID);
    
    % prune lines
    lines_to_prune = false(size(text_cell, 1), 1);
    for i_line = 1 : size(text_cell, 1)
        this_line = text_cell{i_line};
        
        % remove initial white space
        while ~isempty(this_line) && (this_line(1) == ' ' || double(this_line(1)) == 9)
            this_line(1) = [];
        end
        trial_settings_cell{i_line} = this_line; %#ok<AGROW>
        
        % remove comments
        if length(this_line) > 1 && any(ismember(this_line, '#'))
            lines_to_prune(i_line) = true;
        end
        % flag lines consisting only of white space
        if all(ismember(this_line, ' ') | double(this_line) == 9)
            lines_to_prune(i_line) = true;
        end
    end
    trial_settings_cell(lines_to_prune) = [];
    
    number_of_folders = size(trial_settings_cell,2);
    
    for this_folder = 1:number_of_folders
        this_trial_map = strsplit(trial_settings_cell{this_folder}, ',');
        this_trial_scenario = this_trial_map{2};
        
%         if isfile(fullfile(matlab_path,['abm_leftseat_',this_trial_scenario,'.mat']))
%             load(fullfile(matlab_path,['abm_leftseat_',this_trial_scenario,'.mat']));
%             eeg_leftseat = [abm_leftseat.F3, abm_leftseat.F4, abm_leftseat.Fz, ...
%                 abm_leftseat.C3, abm_leftseat.C4, abm_leftseat.Cz,  ...
%                 abm_leftseat.P3, abm_leftseat.P4, abm_leftseat.POz];
%             save_mat_file_name = ['eeg_leftseat_',this_trial_scenario,'.mat'];
%             save([matlab_path filesep save_mat_file_name], 'eeg_leftseat');
%             
%             ekg_leftseat = [abm_leftseat.ECG];
%             save_mat_file_name = ['ekg_leftseat_',this_trial_scenario,'.mat'];
%             save([matlab_path filesep save_mat_file_name], 'ekg_leftseat');
%             
%             EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
%             %import
%             EEG = pop_importdata('dataformat','matlab','nbchan',9,'data',fullfile(data_dir,matlab_path,save_mat_file_name),'srate',256,'pnts',0,'xmin',0);
%             EEG=pop_chanedit(EEG, 'lookup','C:\\Users\\tfettrow\\Documents\\GitHub\\eeglab_current\\eeglab2022.0\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc','changefield',{1,'labels','F3'},'insert',1,'changefield',{1,'datachan',1},'insert',2,'changefield',{3,'labels',''},'insert',3,'changefield',{3,'datachan',1},'changefield',{2,'datachan',1},'delete',3,'delete',2,'delete',1,'append',1,'changefield',{2,'datachan',1},'append',2,'changefield',{3,'datachan',1},'append',3,'changefield',{4,'datachan',1},'append',4,'changefield',{5,'datachan',1},'append',5,'changefield',{6,'datachan',1},'append',6,'changefield',{7,'datachan',1},'append',7,'changefield',{8,'datachan',1},'append',8,'changefield',{9,'datachan',1},'changefield',{1,'labels','F3'},'changefield',{2,'labels','F4'},'changefield',{3,'labels','Fz'},'changefield',{4,'labels','C3'},'changefield',{5,'labels','C4'},'changefield',{6,'labels','Cz'},'changefield',{7,'labels','P3'},'changefield',{8,'labels','P4'},'changefield',{9,'labels','Pz'},'lookup','C:\\Users\\tfettrow\\Documents\\GitHub\\eeglab_current\\eeglab2022.0\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc');
%             
%             % filter
%             EEG = pop_eegfilt( EEG, 1, 40, [], [0], 0, 0, 'fir1', 1);
%             
%             % Clean/Remove Data
%             orig_eeg_size = size(EEG.data);
%             EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',10,'ChannelCriterion',0.86,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion','off','BurstRejection','off','Distance','Euclidian','channels',{'F3','F4','Fz','C3','C4','Cz','P3','P4','Pz'},'availableRAM_GB',8);
%             clean_eeg_size = size(EEG.data);
%             data_removed = orig_eeg_size - clean_eeg_size;
%             eeg_leftseat_channels_removed(this_folder) = data_removed(1);
%             eeg_leftseat_data_points_removed(this_folder) = data_removed(2);
%             
%             legend_cell{this_folder} = this_trial_scenario;
%             
%             
%             % ICA
%             EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
% %             pop_selectcomps(EEG, [1:9] );
%             EEG = pop_iclabel(EEG, 'default');
%             
%             figure; pop_spectopo(EEG, 1, [0      EEG.times(end)], 'EEG' , 'percent', 50, 'freq', [6 12 23], 'freqrange',[2 50],'electrodes','off');
%             fullfilename = fullfile(crew_ids{this_crew},'figures',['eeg_leftseat_frequency_components_',this_trial_scenario]);
%             saveas(gca,fullfilename,'tif');
%             close all
%             
%             pop_viewprops(EEG,0,1:size(EEG.icaweights,2),'freqrange', [1 40],'ICLabel')
%             fullfilename = fullfile(crew_ids{this_crew},'figures',['eeg_leftseat_ica_labels_',this_trial_scenario]);
%             saveas(gca,fullfilename,'tif');
%             close all
          
%             for this_channel = 1:size(EEG.icaweights,2) % using as proxy for # of chans bc TF is lazy (might not always work)
%                 figure; pop_newtimef( EEG, 1, this_channel, [200000   300000], [3         0.8] , 'baseline',[0], 'freqs', [1 50], 'plotitc' , 'off', 'plotphase', 'off', 'padratio', 1);
%                 fullfilename = fullfile(crew_ids{this_crew},'figures',['eeg_leftseat_chan_',num2str(this_channel),'_freq_',this_trial_scenario]);
%                 saveas(gca,fullfilename,'tif');
%                 close all
%             end
            
%             figure; hold on;
%             plot(abm_leftseat.UserTimeStamp, ekg_leftseat)
            
            
            
%         end
        
        
        
        if isfile(fullfile(matlab_path,['abm_rightseat_',this_trial_scenario,'.mat']))
            load(fullfile(matlab_path,['abm_rightseat_',this_trial_scenario,'.mat']));
           
            eeg_rightseat = [abm_rightseat.F3, abm_rightseat.F4, abm_rightseat.Fz, ...
                abm_rightseat.C3, abm_rightseat.C4, abm_rightseat.Cz,  ...
                abm_rightseat.P3, abm_rightseat.P4, abm_rightseat.POz];
            save_mat_file_name = ['eeg_rightseat_',this_trial_scenario,'.mat'];
            save([matlab_path filesep save_mat_file_name], 'eeg_rightseat');
            
            ekg_rightseat = [abm_rightseat.ECG];
            save_mat_file_name = ['ekg_rightseat_',this_trial_scenario,'.mat'];
            save([matlab_path filesep save_mat_file_name], 'ekg_rightseat');
            
            
            
%             EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
%             %import
%             EEG = pop_importdata('dataformat','matlab','nbchan',9,'data',fullfile(data_dir,matlab_path,save_mat_file_name),'srate',256,'pnts',0,'xmin',0);
%             EEG=pop_chanedit(EEG, 'lookup','C:\\Users\\tfettrow\\Documents\\GitHub\\eeglab_current\\eeglab2022.0\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc','changefield',{1,'labels','F3'},'insert',1,'changefield',{1,'datachan',1},'insert',2,'changefield',{3,'labels',''},'insert',3,'changefield',{3,'datachan',1},'changefield',{2,'datachan',1},'delete',3,'delete',2,'delete',1,'append',1,'changefield',{2,'datachan',1},'append',2,'changefield',{3,'datachan',1},'append',3,'changefield',{4,'datachan',1},'append',4,'changefield',{5,'datachan',1},'append',5,'changefield',{6,'datachan',1},'append',6,'changefield',{7,'datachan',1},'append',7,'changefield',{8,'datachan',1},'append',8,'changefield',{9,'datachan',1},'changefield',{1,'labels','F3'},'changefield',{2,'labels','F4'},'changefield',{3,'labels','Fz'},'changefield',{4,'labels','C3'},'changefield',{5,'labels','C4'},'changefield',{6,'labels','Cz'},'changefield',{7,'labels','P3'},'changefield',{8,'labels','P4'},'changefield',{9,'labels','Pz'},'lookup','C:\\Users\\tfettrow\\Documents\\GitHub\\eeglab_current\\eeglab2022.0\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc');
%             
%             % filter
%             EEG = pop_eegfilt( EEG, 1, 40, [], [0], 0, 0, 'fir1', 1);
%             
%             % Clean/Remove Data
%             orig_eeg_size = size(EEG.data);
%             EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',10,'ChannelCriterion',0.6,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion','off','BurstRejection','off','Distance','Euclidian','channels',{'F3','F4','Fz','C3','C4','Cz','P3','P4','Pz'},'availableRAM_GB',8);
%             clean_eeg_size = size(EEG.data);
%             data_removed = orig_eeg_size - clean_eeg_size;
%             eeg_rightseat_channels_removed(this_folder) = data_removed(1);
%             eeg_rightseat_data_points_removed(this_folder) = data_removed(2);
%             
%             legend_cell{this_folder} = this_trial_scenario;
%             
%             % ICA
%             EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
% %             pop_selectcomps(EEG, [1:9] );
%             EEG = pop_iclabel(EEG, 'default');
            
%             figure; pop_spectopo(EEG, 1, [0      EEG.times(end)], 'EEG' , 'percent', 50, 'freq', [6 12 23], 'freqrange',[2 50],'electrodes','off');
%             fullfilename = fullfile(crew_ids{this_crew},'figures',['eeg_rightseat_frequency_components_',this_trial_scenario]);
%             saveas(gca,fullfilename,'tif');
%             close all
%             
%             pop_viewprops(EEG,0,1:size(EEG.icaweights,2),'freqrange', [1 40],'ICLabel')
%             fullfilename = fullfile(crew_ids{this_crew},'figures',['eeg_rightseat_ica_labels_',this_trial_scenario]);
%             saveas(gca,fullfilename,'tif');
%             close all
%             
%             for this_channel = 1:size(EEG.icaweights,2) % using as proxy for # of chans bc TF is lazy (might not always work)
%                 figure; pop_newtimef( EEG, 1, this_channel, [200000   300000], [3         0.8] , 'baseline',[0], 'freqs', [1 50], 'plotitc' , 'off', 'plotphase', 'off', 'padratio', 1);
%                 fullfilename = fullfile(crew_ids{this_crew},'figures',['eeg_leftseat_chan_',num2str(this_channel),'_freq_',this_trial_scenario]);
%                 saveas(gca,fullfilename,'tif');
%             end

            figure; hold on;
            plot(abm_rightseat.UserTimeStamp, ekg_rightseat)
            set(gcf, 'ToolBar', 'none');
            set(gcf, 'MenuBar', 'none');
            set(get(gca, 'xlabel'), 'visible', 'off');
            set(get(gca, 'ylabel'), 'visible', 'off');
            set(get(gca, 'title'), 'visible', 'off');
            legend(gca, 'hide');
            saveas(gcf,string(path_to_project, "/Figures/", 'smarteyeGaze_',crew_ids(i_crew), '.tif'))
        end
       
    end
%     figure; hold on;
    eeg_leftseat_cleaning_stats_table = table;
    eeg_leftseat_cleaning_stats_table.scenarios = legend_cell';
    eeg_leftseat_cleaning_stats_table.channels_removed = eeg_leftseat_channels_removed';
    eeg_leftseat_cleaning_stats_table.datapoints_removed = eeg_leftseat_data_points_removed';
    save_mat_file_name = ['eeg_leftseat_cleaning_stats_table.mat'];
    save([matlab_path filesep save_mat_file_name], 'eeg_leftseat_cleaning_stats_table');
    
    eeg_rightseat_cleaning_stats_table = table;
    eeg_rightseat_cleaning_stats_table.scenarios = legend_cell';
    eeg_rightseat_cleaning_stats_table.channels_removed = eeg_rightseat_channels_removed';
    eeg_rightseat_cleaning_stats_table.datapoints_removed = eeg_rightseat_data_points_removed';
    save_mat_file_name = ['eeg_rightseat_cleaning_stats_table.mat'];
    save([matlab_path filesep save_mat_file_name], 'eeg_rightseat_cleaning_stats_table');
    
%     eeg_cleaning_stats_table = table(eeg_leftseat_data_points_removed',eeg_leftseat_channels_removed');
%     eeg_leftseat_data_points_removed
%     eeg_leftseat_channels_removed
%     legend_cell
%     (this_folder)
end

