clear,clc
crew_ids = {'Crew_13'};
% crew_ids = {'Crew_02','Crew_03','Crew_04','Crew_05','Crew_06','Crew_07','Crew_08','Crew_09','Crew_10','Crew_11','Crew_12','Crew_13'};
% file_types = {'abm_leftseat','abm_rightseat'};
scenarios = {'1','2','3','5','6','7'};
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA';
file_types = {"smarteye_leftseat","smarteye_rightseat"};
% set(groot, 'defaultAxesTickLabelInterpreter','none'); set(groot, 'defaultLegendInterpreter','none');

for i_crew = 1:length(crew_ids)
    for i_scenario = 1:length(scenarios)
        fh = figure; hold on;
        fh.WindowState = 'maximized';
%         subplot(6,1,i_scenario)
%         hold on;
		for i_seat = 1:length(file_types)
           
           crew_dir = fullfile(path_to_project,crew_ids(i_crew));
           process_dir_name = fullfile(crew_dir, "/Processing/");

           smarteye_data = readtable(fullfile(process_dir_name,strcat(file_types{i_seat},'_scenario',scenarios{i_scenario},'.csv')));

            direction_gaze = [smarteye_data.GazeDirectionX(2:end),smarteye_data.GazeDirectionY(2:end),smarteye_data.GazeDirectionZ(2:end)];
            origin_gaze = [smarteye_data.GazeOriginX(2:end),smarteye_data.GazeOriginY(2:end),smarteye_data.GazeOriginZ(2:end)];
            quality_gaze = [smarteye_data.GazeDirectionQ(2:end)];
%             origin_gaze = [ox, oy, oz];
%             origin_head = [hx,hy, hz];
%             origin_lefteye = [lex, ley, lez];
%             origin_righteye = [rex, rey, rez];
            
            mag = (sqrt(direction_gaze(:,1).^2 + direction_gaze(:,2).^2 + direction_gaze(:,3).^2)/1);

            good_indices = find((mag ~= 0) & (quality_gaze >=.6));
            q(:,:) = sphere_stereograph(direction_gaze(good_indices,:)');
            
            if (i_seat == 1)
                scatter(q(1,:), q(2,:),6, 'MarkerFaceColor','b');
                alpha(.1);
                
                text(-.9,-.1,string(round(length(good_indices) / length(direction_gaze)*100)),'Color','b', 'FontSize', 24)
            else
                scatter(q(1,:), q(2,:),6, 'MarkerFaceColor','r');
                alpha(.1);
                text(.9,-.1,string(round(length(good_indices) / length(direction_gaze)*100)),'Color','r', 'FontSize', 24)
            end
             clear q mag
        end
         xlim([-1 1]);
         ylim([-1 0]);
         set(gcf, 'ToolBar', 'none');
         set(gcf, 'MenuBar', 'none');
         set(get(gca, 'xlabel'), 'visible', 'off');
         set(get(gca, 'ylabel'), 'visible', 'off');
         set(get(gca, 'title'), 'visible', 'off');
         legend(gca, 'hide');
         set(gca,'visible','off')
%          ylabel(scenarios(i_scenario),'FontSize',24)
         fullfilename = fullfile(path_to_project,crew_ids{i_crew}, "Figures",strcat("smarteyeGaze_",scenarios{i_scenario}));
         saveas(gca,fullfilename,'tiff');
         close all;
%          saveas(gcf,fullfile(path_to_project, "Figures",["smarteyeGaze_",crew_ids(i_crew), '.tif']))
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
% for i =1:length(dx)    
% %     direction_normed(i,:) = normVector([dx(i),dy(i),dz(i)]);
%     
% % sqrt(dx(i)^2 + dy(i)^2 + dz(i)^2)/1;
% end


% figure; hold on;
% for i = 1 : length(direction_gaze)
%     hold on;
%     mag(i) = (sqrt(direction_gaze(i,1)^2 + direction_gaze(i,2)^2 + direction_gaze(i,3)^2)/1);
%     
%     if (mag(i)~= 0)
%         q(:,i) = sphere_stereograph(direction_gaze(i,:)');
%         plot(q(1,i), q(2,i), '.');
% % %         plot3(direction_gaze(i,1),direction_gaze(i,2),direction_gaze(i,3),'.');
%     end
%     xlim(
% %     quiver3(origin_gaze(i_index,1), oy(i_index,2), oz(i_index,3), direction_normed(i_index, 1), direction_normed(i_index,2), direction_normed(i_index,3)); 
% end
% plot(norm_vector(
% asdf - 1;

%  number_of_indices = length(smarteyeleftseatscenario1.GazeDirectionX);
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