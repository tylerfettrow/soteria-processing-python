

number_of_indices = length(smarteyeleftseatscenario1.GazeDirectionX);

figure; hold on;
for i_index = 2 : number_of_indices
%     plot3(smarteyeleftseatscenario1.GazeDirectionX(i_index),smarteyeleftseatscenario1.GazeDirectionY(i_index),smarteyeleftseatscenario1.GazeDirectionZ(i_index),'*')
%     plot3(smarteyeleftseatscenario1.GazeOriginX(i_index),smarteyeleftseatscenario1.GazeOriginY(i_index),smarteyeleftseatscenario1.GazeOriginZ(i_index),'o')
    quiver3(smarteyeleftseatscenario1.GazeOriginX(i_index),smarteyeleftseatscenario1.GazeOriginY(i_index),smarteyeleftseatscenario1.GazeOriginZ(i_index),smarteyeleftseatscenario1.GazeDirectionX(i_index),smarteyeleftseatscenario1.GazeDirectionY(i_index),-smarteyeleftseatscenario1.GazeDirectionZ(i_index))
    pause(.1);
end