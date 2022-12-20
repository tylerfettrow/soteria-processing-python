% check_synched_vs_raw

clear,clc
crew_ids = {'Crew1'};
% crew_ids = {'Crew2','Crew3','Crew4','Crew5','Crew6'};
% file_types = {'abm_leftseat','abm_rightseat'};
% scenarios = {'1','2','3','5','6','7','8','9'};
filename_date = '2022.05.03_09.59.59';
data_dir = pwd;
% set(groot, 'defaultAxesTickLabelInterpreter','none'); set(groot, 'defaultLegendInterpreter','none');

for this_crew = 1:length(crew_ids)
    figure; hold on;
    if isfile(fullfile(crew_ids{this_crew},'Synched',filename_date,'ABM.log'))
        abm_leftseat_synched = readtable(fullfile(crew_ids{this_crew},'Synched',filename_date,'ABM.log'));
        abm_leftseat_synched = adjust_timestamps(abm_leftseat_synched);
        abm_leftseat_synched_time = abm_leftseat_synched.UserTimeStamp;
        plot(abm_leftseat_synched_time)
    end
    if isfile(fullfile(crew_ids{this_crew},'Raw','soteria1_abm',filename_date,[filename_date, '.log']))
        abm_leftseat_raw = readtable(fullfile(crew_ids{this_crew},'Raw','soteria1_abm',filename_date,[filename_date, '.log']));
        abm_leftseat_raw = adjust_timestamps(abm_leftseat_raw);
        abm_leftseat_raw_time = abm_leftseat_raw.UserTimeStamp;
        plot(abm_leftseat_raw_time)
    end
%     time_vec_diff = diff(abm_leftseat_synched.UserTimeStamp, abm_leftseat_raw.UserTimeStamp)
end