% save physio

% need a settings file that maps trial start_times to scenario_number


clear file_name_list;
data_dir = pwd;
trial_folders = dir(fullfile(data_dir,'Synched'));
% [folder_name_list{1:length(trial_folders)}] = deal(trial_folders.name);
% number_of_folders = length(folder_name_list);

% trialmap_settings_file = '
fileID = fopen('trial_settings.txt', 'r');
% levels_back_subject = levels_back_subject + 1;
% levels_back_task = levels_back_task + 1;
% disp('searching for outlier_removal_settings...')

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

for this_line = 1:number_of_folders
    this_trial_map = strsplit(trial_settings_cell{this_line}, ',');
    this_synced_folder_name = this_trial_map{1};
    this_trial_scenario = this_trial_map{2};
    
    save_folder = 'matlab';
    if ~isfolder('matlab')
        mkdir('matlab')
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'ABM.log'))
        abm_leftseat = readtable(fullfile('Synched',this_synced_folder_name,'ABM.log'));
        if ~isempty(abm_leftseat)
            abm_leftseat = adjust_timestamps(abm_leftseat);
            save_file_name = ['abm_leftseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'abm_leftseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'ABM-1.log'))
        abm_rightseat = readtable(fullfile('Synched',this_synced_folder_name,'ABM-1.log'));
        if ~isempty(abm_rightseat)
            abm_rightseat = adjust_timestamps(abm_rightseat);
            save_file_name = ['abm_rightseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'abm_rightseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Acc.log'))
        emp_acc_leftseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Acc.log'));
        if ~isempty(emp_acc_leftseat)
            emp_acc_leftseat = adjust_timestamps(emp_acc_leftseat);
            save_file_name = ['emp_acc_leftseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_acc_leftseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Bvp.log'))
        emp_bvp_leftseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Bvp.log'));
        if ~isempty(emp_bvp_leftseat)
            emp_bvp_leftseat = adjust_timestamps(emp_bvp_leftseat);
            save_file_name = ['emp_bvp_leftseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_bvp_leftseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Gsr.log'))
        emp_gsr_leftseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Gsr.log'));
        if ~isempty(emp_gsr_leftseat)
            emp_gsr_leftseat = adjust_timestamps(emp_gsr_leftseat);
            save_file_name = ['emp_gsr_leftseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_gsr_leftseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Ibi.log'))
        emp_ibi_leftseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Ibi.log'));
        if ~isempty(emp_ibi_leftseat)
            emp_ibi_leftseat = adjust_timestamps(emp_ibi_leftseat);
            save_file_name = ['emp_ibi_leftseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_ibi_leftseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Temp.log'))
        emp_temp_leftseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device_Temp.log'));
        if ~isempty(emp_temp_leftseat)
            emp_temp_leftseat = adjust_timestamps(emp_temp_leftseat);
            save_file_name = ['emp_temp_leftseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_temp_leftseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Acc.log'))
        emp_acc_rightseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Acc.log'));
        if ~isempty(emp_temp_leftseat)
            emp_acc_rightseat = adjust_timestamps(emp_acc_rightseat);
            save_file_name = ['emp_acc_rightseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_acc_rightseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Bvp.log'))
        emp_bvp_rightseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Bvp.log'));
        if ~isempty(emp_temp_leftseat)
            emp_bvp_rightseat = adjust_timestamps(emp_bvp_rightseat);
            save_file_name = ['emp_bvp_rightseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_bvp_rightseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Gsr.log'))
        emp_gsr_rightseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Gsr.log'));
        if ~isempty(emp_temp_leftseat)
            emp_gsr_rightseat = adjust_timestamps(emp_gsr_rightseat);
            save_file_name = ['emp_gsr_rightseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_gsr_rightseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Ibi.log'))
        emp_ibi_rightseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Ibi.log'));
        if ~isempty(emp_temp_leftseat)
            emp_ibi_rightseat = adjust_timestamps(emp_ibi_rightseat);
            save_file_name = ['emp_ibi_rightseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_ibi_rightseat');
        end
    end
    
    if isfile(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Temp.log'))
        emp_temp_rightseat = readtable(fullfile('Synched',this_synced_folder_name,'Emp_Emp_Device2_Temp.log'));
        if ~isempty(emp_temp_leftseat)
            emp_temp_rightseat = adjust_timestamps(emp_temp_rightseat);
            save_file_name = ['emp_temp_rightseat_',this_trial_scenario,'.mat'];
            save([save_folder filesep save_file_name], 'emp_temp_rightseat');
        end
    end
    
     

    disp(['saved data from trial: ', string(this_line), '  scenario:', string(this_trial_scenario)])

     
     
%      if save_figures
%          saveas(gca, filename, 'png')
%      end
     
     
end