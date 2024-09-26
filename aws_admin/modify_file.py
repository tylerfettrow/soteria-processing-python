import pandas as pd
import importlib
from argparse import ArgumentParser
import sys
sys.path.insert(1, '/home/tfettrow/Documents/soteria_code/python4GSlocal/')
import helpers
import os
import pdb

importlib.reload(helpers)

helper = helpers.HELP()

###############################################
parser = ArgumentParser(description='Export Raw Data')
parser.add_argument('-c', '--crews', type=str, default=['01','02','03','04','05','06','07','08','09','10','11','13'])
parser.add_argument('-s', '--scenarios', type=str, default=["1", "2", "3", "5", "6", "7"])
parser.add_argument('-d', '--directory', type=str)
parser.add_argument('--Push2Cloud', type=bool, default=False)
args = parser.parse_args()
crew_arr,scenarios = helper.ParseCrewAndScenario(sys.argv, args.crews,args.scenarios)
if args.directory:
    helper.local_process_dir = args.directory
###############################################

for i_crew in range(len(crew_arr)):
    helper.crew_dir = "Crew_"+crew_arr[i_crew]+"/"
    print(helper.crew_dir)
    trial_settings = helper.read_bucket_table("trial_settings.txt")
    new_trial_settings = trial_settings.drop([6,7])

    output_dir = helper.local_process_dir + helper.crew_dir 
    os.makedirs(output_dir, exist_ok=True)
    # breakpoint()
    new_trial_settings.to_csv(output_dir +"trial_settings" + ".txt", index=False)
    # breakpoint()
    # helper.sync_crew_folder_storage()