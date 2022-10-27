import os
import shutil


######### BOX #################################################
#crews_to_process = ['Crew1', 'Crew2', 'Crew3', 'Crew4', 'Crew5', 'Crew6', 'Crew7', 'Crew8', 'Crew9', 'Crew10', 'Crew11', 'Crew12', 'Crew13']
# crews_to_process = ['Crew_02', 'Crew_03', 'Crew_04', 'Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
crews_to_process = ['Crew_01']
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'

for this_crew in crews_to_process:
	crew_dir = path_to_project + '/' + this_crew

	print(crew_dir)
	#trial_folders = os.listdir(crew_dir + '/Synched')
	#os.rename(crew_dir + '/Analysis',crew_dir + '/Processing')

	#shutil.rmtree(crew_dir + '/Analysis/', ignore_errors=True)
######### GSUTIL #################################################