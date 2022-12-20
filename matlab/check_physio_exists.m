% Pyshio QA
% expected to be run within study folder ("SOTERIA")

% DATA PRESENT
% for each file within each trial
% 1) determine whether there is data present for every emp folder
% 2) generate heat map of whether data present (green good red bad)
% % % we know what there "should be", so ok to hard code that
% 3) save the heatmap fig in "figures"

clear,clc
% crew_ids = {'Crew1','Crew2','Crew3','Crew4','Crew5','Crew6'};
crew_ids = {'Crew7','Crew8'};
file_types = {'abm_leftseat','abm_rightseat','emp_acc_leftseat','emp_acc_rightseat','emp_bvp_leftseat','emp_bvp_rightseat','emp_gsr_leftseat','emp_gsr_rightseat','emp_ibi_leftseat','emp_ibi_rightseat','emp_temp_leftseat','emp_temp_rightseat'};
scenarios = {'1','2','3','5','6','7','8','9'};
data_dir = pwd;
set(groot, 'defaultAxesTickLabelInterpreter','none'); set(groot, 'defaultLegendInterpreter','none');

for this_crew = 1:length(crew_ids)
    for i_scenario = 1 : length(scenarios)
        for i_devicefile = 1:length(file_types)
            if isfile(fullfile(crew_ids{this_crew},'matlab',[file_types{i_devicefile},'_scenario',scenarios{i_scenario},'.mat']))
                file_existence_matrix(i_devicefile, i_scenario) = 1;
            else
                file_existence_matrix(i_devicefile, i_scenario) = 0;
            end
        end
    end
    
    figure; hold on;
%     grid on
    imagesc(([1:8])+0.5, (1:12)+0.5, file_existence_matrix);          % Plot the image
    colormap(winter);                              % Use a gray colormap
    % custom grid
    for row = 0 : 13
        line([1, 9], [row, row], 'Color', 'k');
    end
    for col = 1 : 9
        line([col, col], [1, 13], 'Color', 'k');
    end
    xlabel('scenario number')
    set(gca,'XTick', 1.5:8.5,'XTickLabel',scenarios)
    ylabel('file name')
    set(gca,'YTick', 1.5:12.5,'YTickLabel',file_types)
    ylim([1 13])
    title(['File Existence Heatmap ',crew_ids{this_crew}])
    mkdir(fullfile(crew_ids{this_crew},'figures'))
    fullfilename = fullfile(crew_ids{this_crew},'figures',['file_existence_heatmap_',crew_ids{this_crew}]);
    saveas(gca,fullfilename,'tiff');
    
    save_file_name = fullfile(crew_ids{this_crew},'matlab','file_existence_matrix.mat');
    save(save_file_name, 'file_existence_matrix','scenarios','file_types');
    
end

