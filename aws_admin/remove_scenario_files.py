import pandas as pd
import helpers
import importlib
from argparse import ArgumentParser
import sys
import subprocess
import glob
import os
importlib.reload(helpers)

helper = helpers.HELP()

###############################################
parser = ArgumentParser(description='Export Raw Data')
parser.add_argument('-c', '--crews', type=str, default=['01','02','03','04','05','06','07','08','09','10','11','13'])
parser.add_argument('-s', '--scenarios', type=str, default=["8", "9"])
parser.add_argument('-d', '--directory', type=str)
parser.add_argument('--Push2Cloud', type=bool, default=False)
args = parser.parse_args()
crew_arr,scenarios = helper.ParseCrewAndScenario(sys.argv, args.crews,args.scenarios)
if args.directory:
    helper.local_process_dir = args.directory
###############################################

for i_crew in range(len(crew_arr)):
	helper.crew_dir = "Crew_"+crew_arr[i_crew]+"/"
	
	# cd *
	# breakpoint()
	for f in glob.glob("/home/tfettrow/nasa-soteria-data/"+helper.crew_dir+"Processing/*scenario8.csv"):
		print(f)
		os.remove(f)

	for f in glob.glob("/home/tfettrow/nasa-soteria-data/"+helper.crew_dir+"Processing/*scenario9.csv"):
		print(f)
		os.remove(f)	
	

	# for i_scenario in range(len(scenarios)):

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/abm_leftseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# print(cmd)
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "Processing/abm_leftseat_scenario" + scenarios[i_scenario]+".csv")

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_acc_leftseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + " emp_acc_leftseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_bvp_leftseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + " emp_bvp_leftseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_gsr_leftseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "emp_gsr_leftseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_ibi_leftseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "emp_ibi_leftseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_temp_leftseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "emp_temp_leftseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/smarteye_leftseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "smarteye_leftseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/abm_rightseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "abm_rightseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_acc_rightseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "emp_acc_rightseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_bvp_rightseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "emp_bvp_rightseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_gsr_rightseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "emp_gsr_rightseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_ibi_rightseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "emp_ibi_rightseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/emp_temp_rightseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "emp_temp_rightseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/smarteye_rightseat_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "smarteye_rightseat_scenario" + scenarios[i_scenario])

		# cmd = 'aws s3 rm s3://nasa-soteria-data/{}Processing/ifd_cockpit_scenario{}.csv'.format(helper.crew_dir,scenarios[i_scenario])
		# subprocess.call(cmd,shell=True)
		# # print("removing:" + helper.crew_dir + "ifd_cockpit_scenario" + scenarios[i_scenario])

    # print("args.Push2Cloud = " + str(args.Push2Cloud))
    # if args.Push2Cloud:
    #     helper.sync_crew_folder_storage(crew_dir)
