
ext_root = fullfile('D:',filesep,'SOTERIA',filesep,'SOTERIA_Study_Data');
box_root = fullfile('C:',filesep,'Users','tfettrow',filesep,'Box',filesep,'SOTERIA'); % thought this might work for mac but after a quick thought it prob wont

crew_id = 'Crew_11';

% RAW 
% mkdir(fullfile(box_root,crew_id,'Raw'));
% copyfile(fullfile(ext_root,crew_id,'Raw',filesep,'abm_soteria1'), fullfile(box_root,crew_id,'Raw',filesep,'abm_soteria1'))
% copyfile(fullfile(ext_root,crew_id,'Raw',filesep,'abm_soteria2'), fullfile(box_root,crew_id,'Raw',filesep,'abm_soteria2'))
% copyfile(fullfile(ext_root,crew_id,'Raw',filesep,'smarteye_leftseat'), fullfile(box_root,crew_id,'Raw',filesep,'smarteye_leftseat'))
% copyfile(fullfile(ext_root,crew_id,'Raw',filesep,'smarteye_rightseat'), fullfile(box_root,crew_id,'Raw',filesep,'smarteye_rightseat'))
% copyfile(fullfile(ext_root,crew_id,'Raw',filesep,'ifd'), fullfile(box_root,crew_id,'Raw',filesep,'ifd'))



% SYNCHED 
% mkdir(fullfile(box_root,crew_id,'Synched'));
% copyfile(fullfile(ext_root,crew_id,'Synched'), fullfile(box_root,crew_id,'Synched'))
folderlist = dir(fullfile(ext_root,crew_id,'Synched'));
% i_folder = 7:length(folderlist)
for i_folder = 3:length(folderlist)
    filelist = dir(fullfile(ext_root,crew_id,'Synched',folderlist(i_folder).name));
 
%    
    % Exclude Cameras
%     camera_prefix = [".","..","inst" , "scenecamera"];
%     filelist = filelist(~startsWith({filelist.name}, camera_prefix));
% %     Include Cameras
    camera_prefix = ["inst" , "scenecamera"];
    filelist = filelist(startsWith({filelist.name}, camera_prefix));
   
    for i_file = 1:length(filelist)
        trial_foldername = strsplit(filelist(i_file).folder,'\');
        if ~isfolder(fullfile(box_root,crew_id,'Synched', trial_foldername{end}))
            mkdir(fullfile(box_root,crew_id,'Synched', trial_foldername{end}));
        end
        copyfile(fullfile(filelist(i_file).folder, filelist(i_file).name),fullfile(box_root,crew_id,'Synched', trial_foldername{end},filelist(i_file).name),'f');
        disp(['copying ', filelist(i_file).name, ' to ', fullfile(box_root,crew_id,'Synched',trial_foldername{end})])
    end
end
