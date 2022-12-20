% check physio_exists_summary
clear,clc

crew_ids = {'Crew1','Crew2','Crew3','Crew4','Crew5','Crew6','Crew7','Crew8'};
set(groot, 'defaultAxesTickLabelInterpreter','none'); set(groot, 'defaultLegendInterpreter','none');

for this_crew = 1:length(crew_ids)
    
    load(fullfile(crew_ids{this_crew},'matlab','file_existence_matrix.mat'));
    
    total_file_existence_matrix(:,:,this_crew) = file_existence_matrix;
end

total_file_existence_matrix_percent = squeeze(sum(total_file_existence_matrix,3))./ size(total_file_existence_matrix,3) .* 100; %percent
heatmap(total_file_existence_matrix_percent)

file_types_latex = insertBefore(file_types,"_","\");

title(['Percent of physio files available (', num2str(length(crew_ids)), ' crews)'] )
set(gca,'XLabel','scenario label','XDisplayLabels',scenarios)
set(gca,'YLabel','file name','YDisplayLabels',file_types_latex)

mkdir(fullfile('figures'))
fullfilename = fullfile('figures','file_existence_heatmap');
saveas(gca,fullfilename,'tif');