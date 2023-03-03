clear,clc
crew_ids = {'Crew_01'};
% crew_ids = {'Crew2','Crew3','Crew4','Crew5','Crew6'};
% file_types = {'abm_leftseat','abm_rightseat'};
scenarios = {'1','2','3','5','6','7'};
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA';
file_types = {"smarteye_leftseat","smarteye_rightseat"}
% set(groot, 'defaultAxesTickLabelInterpreter','none'); set(groot, 'defaultLegendInterpreter','none');

for i_crew = 1:length(crew_ids)
    for i_scenario = 1:length(scenarios)
		for i_seat = 1:length(file_types)
           crew_dir = fullfile(path_to_project,crew_ids(i_crew));
           process_dir_name = fullfile(crew_dir, "/Processing/");

           smarteye_data = readtable(fullfile(process_dir_name,strcat(file_types{i_seat},'_scenario',scenarios{i_scenario},'.csv')));
%            dx = smarteyeleftseatscenario1.GazeDirectionX(2:end);
%            dy = smarteyeleftseatscenario1.GazeDirectionY(2:end);
%            dz = smarteyeleftseatscenario1.GazeDirectionZ(2:end);
%            ox = smarteyeleftseatscenario1.GazeOriginX(2:end);
%            oy = smarteyeleftseatscenario1.GazeOriginY(2:end);
%            oz = smarteyeleftseatscenario1.GazeOriginZ(2:end);
%            hx = smarteyeleftseatscenario1.HeadPosX(2:end);
%            hy = smarteyeleftseatscenario1.HeadPosY(2:end);
%            hz = smarteyeleftseatscenario1.HeadPosZ(2:end);
%            lex = smarteyeleftseatscenario1.LeftEyePosX(2:end);
%            ley = smarteyeleftseatscenario1.LeftEyePosY(2:end);
%            lez = smarteyeleftseatscenario1.LeftEyePosZ(2:end);
%            rex = smarteyeleftseatscenario1.RightEyePosX(2:end);
%            rey = smarteyeleftseatscenario1.RightEyePosY(2:end);
%            rez = smarteyeleftseatscenario1.RightEyePosZ(2:end);

            direction_gaze = [dx, dy, dz];
            origin_gaze = [ox, oy, oz];
            origin_head = [hx,hy, hz];
            origin_lefteye = [lex, ley, lez];
            origin_righteye = [rex, rey, rez];
            
            

        end
    end
end



% figure;
% for i = 1:length(dx) 
%     hold on;
%     plot3(origin_head(i,1),origin_head(i,2),origin_head(i,3),'o','markersize',1000,'MarkerFaceColor','g');
%     plot3(origin_gaze(i,1),origin_gaze(i,2),origin_gaze(i,3),'o','markersize',500),'MarkerFaceColor','k';
%     plot3(origin_lefteye(i,1),origin_lefteye(i,2),origin_lefteye(i,3),'^','markersize',12,'MarkerFaceColor','b');
%     plot3(origin_righteye(i,1),origin_righteye(i,2),origin_righteye(i,3),'^','markersize',12,'MarkerFaceColor','r');
%     clf;
% end

% figure; hold on;
for i =1:length(dx)    
%     direction_normed(i,:) = normVector([dx(i),dy(i),dz(i)]);
    
% sqrt(dx(i)^2 + dy(i)^2 + dz(i)^2)/1;
end


% figure; hold on;
for i = 1 : length(direction_gaze)
    hold on;
    mag(i) = (sqrt(direction_gaze(i,1)^2 + direction_gaze(i,2)^2 + direction_gaze(i,3)^2)/1);
    
    if (mag(i)~= 0)
        q(:,i) = sphere_stereograph(direction_gaze(i,:)');
        plot(q(1,i), q(2,i), '.');
% %         plot3(direction_gaze(i,1),direction_gaze(i,2),direction_gaze(i,3),'.');
    end
    xlim(
%     quiver3(origin_gaze(i_index,1), oy(i_index,2), oz(i_index,3), direction_normed(i_index, 1), direction_normed(i_index,2), direction_normed(i_index,3)); 
end
% plot(norm_vector(
% asdf - 1;

 number_of_indices = length(smarteyeleftseatscenario1.GazeDirectionX);
% 
% figure; hold on;
% for i_index = 2 : 25
% %     plot3(smarteyerightseatscenario1.GazeDirectionX(i_index),smarteyerightseatscenario1.GazeDirectionY(i_index),smarteyerightseatscenario1.GazeDirectionZ(i_index),'*')
% %     plot3(smarteyerightseatscenario1.GazeOriginX(i_index),smarteyerightseatscenario1.GazeOriginY(i_index),smarteyerightseatscenario1.GazeOriginZ(i_index),'o')
%     quiver3(smarteyerightseatscenario1.GazeOriginX(i_index),smarteyerightseatscenario1.GazeOriginY(i_index),smarteyerightseatscenario1.GazeOriginZ(i_index),smarteyerightseatscenario1.GazeDirectionX(i_index),smarteyerightseatscenario1.GazeDirectionY(i_index),-smarteyerightseatscenario1.GazeDirectionZ(i_index))
%     pause(.025);
% end

% project vectors onto unit vector with eye as zero initially
% convert sphere to plane