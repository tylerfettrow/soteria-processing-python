import numpy as np
import os
from os.path import exists
import matplotlib.pyplot as plt
import pandas as pd
import math
import subprocess
import time
import shutil
import helpers
import importlib
from argparse import ArgumentParser
import sys

## WARNING: events subject to change pending expert (LOSA/LIT) review of the DVR recordings

importlib.reload(helpers)

helper = helpers.HELP()

###############################################
parser = ArgumentParser(description="Export Raw Data")
parser.add_argument(
    "-c", "--crews", type=str, default=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "13"]
)
parser.add_argument("-s", "--scenarios", type=str, default=["1", "2", "3", "5", "6", "7"])
parser.add_argument("-d", "--directory", type=str)
parser.add_argument("--Push2Cloud", type=bool, default=False)
args = parser.parse_args()
crew_arr, scenarios = helper.ParseCrewAndScenario(sys.argv, args.crews, args.scenarios)
if args.directory:
    helper.local_process_dir = args.directory
#######################################

for i_crew in range(len(crew_arr)):
    event_vector_timesec = np.zeros((2, len(scenarios)))
    helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"
    process_dir_name = helper.crew_dir + "Processing/"

    for i_scenario in range(len(scenarios)):
        print("finding events: " + process_dir_name + "scenario" + scenarios[i_scenario])

        if scenarios[i_scenario] == "1":
            event_vector_timesec[:, i_scenario] = [130, 865]
        if scenarios[i_scenario] == "2":
            event_vector_timesec[:, i_scenario] = [135, 600]
        if scenarios[i_scenario] == "3":
            # - tailwind (speed)
            # - autopilot A disengage
            # event_vector_timesec[:,i_scenario] = np.array([
            event_1_idx = 0
            # ifd_data = helper.read_bucket_table(process_dir_name + 'ifd_cockpit_scenario' + scenarios[i_scenario] + '.csv')
            ifd_data = helper.read_local_table("Processing/ifd_cockpit_scenario" + scenarios[i_scenario] + ".csv")

            diff_alt = np.diff(ifd_data.altitude)
            for idx in range(0, len(diff_alt)):
                if diff_alt[idx] <= -0.3:
                    event_1_idx = idx
                    break

                event_vector_timesec[:, i_scenario] = [ifd_data.UserTimeStamp[event_1_idx], 540]
        if scenarios[i_scenario] == "5":
            event_1_idx = 0
            event_2_idx = 0
            # ifd_data = helper.read_bucket_table(process_dir_name + 'ifd_cockpit_scenario' + scenarios[i_scenario] + '.csv')
            ifd_data = helper.read_local_table("Processing/ifd_cockpit_scenario" + scenarios[i_scenario] + ".csv")

            roll_dps = ifd_data.roll_angle_rate_dps
            for idx in range(0, len(roll_dps)):
                if roll_dps[idx] <= -10:
                    event_1_idx = idx
                    break

            lat = ifd_data.latitude
            lon = ifd_data.longitude
            lat_t = 34.89083
            lon_t = -80.4755
            lat_diff = (lat - lat_t) * 60  # deg->NM #WARNING: not accounting for sphere
            lon_diff = (lon - lon_t) * 60  # deg->NM #WARNING: not accounting for sphere
            rad_vector = np.zeros((len(lat_diff)))
            for idx in range(0, len(lat_diff)):
                this_rad = math.sqrt(pow(lat_diff[idx], 2) + pow(lon_diff[idx], 2))
                # rad_vector[idx] = math.sqrt(pow(lat_diff[idx], 2) + pow(lon_diff[idx], 2))
                if this_rad <= 10:
                    event_2_idx = idx
                    break

            if helper.crew_dir == "Crew_04":
                event_vector_timesec[:, i_scenario] = [ifd_data.UserTimeStamp[event_1_idx], 0]
            elif helper.crew_dir == "Crew_05":
                event_vector_timesec[:, i_scenario] = [0, 0]
            elif helper.crew_dir == "Crew_13":
                event_vector_timesec[:, i_scenario] = [0, 0]
            else:
                event_vector_timesec[:, i_scenario] = [
                    ifd_data.UserTimeStamp[event_1_idx],
                    ifd_data.UserTimeStamp[event_2_idx],
                ]

        if scenarios[i_scenario] == "6":
            event_vector_timesec[:, i_scenario] = [30, 370]

        if scenarios[i_scenario] == "7":
            event_vector_timesec[:, i_scenario] = [0, 0]

    event_vector_timesec_df = pd.DataFrame(event_vector_timesec)
    helper.store_file(event_vector_timesec_df, "Processing/", "event_vector_scenario")

    # print("args.Push2Cloud = " + str(args.Push2Cloud))
    if args.Push2Cloud:
        helper.sync_crew_folder_storage()
