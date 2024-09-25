import pandas as pd
import helpers
import importlib
from argparse import ArgumentParser
import sys
import os

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
###############################################

for i_crew in range(len(crew_arr)):
    helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"

    trial_settings = helper.read_bucket_table("trial_settings.txt")
    output_dir = helper.local_process_dir + helper.crew_dir
    trial_settings.to_csv(output_dir + "trial_settings" + ".txt", index=False)

    for this_folder in range(trial_settings.shape[0]):
        if os.path.exists(output_dir + "Processing/abm_leftseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(output_dir + "Processing/abm_leftseat_" + trial_settings.Scenario[this_folder] + ".csv already exists")
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: " + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/ABM.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/ABM.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                abm_leftseat = helper.adjust_timestamps(this_table)
                helper.store_file(abm_leftseat, "Processing/", "abm_leftseat_" + str(trial_settings.Scenario[this_folder]))
                print("converted to: abm_leftseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_acc_leftseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(output_dir + "Processing/emp_acc_leftseat_" + trial_settings.Scenario[this_folder] + ".csv already exists")
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Acc.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Acc.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_acc_leftseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_acc_leftseat, "Processing/", "emp_acc_leftseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_acc_leftseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_bvp_leftseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(output_dir + "Processing/emp_bvp_leftseat_" + trial_settings.Scenario[this_folder] + ".csv already exists")
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Bvp.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Bvp.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_bvp_leftseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_bvp_leftseat, "Processing/", "emp_bvp_leftseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_bvp_leftseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_gsr_leftseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(output_dir + "Processing/emp_gsr_leftseat_" + trial_settings.Scenario[this_folder] + ".csv already exists")
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Gsr.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Gsr.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_gsr_leftseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_gsr_leftseat, "Processing/", "emp_gsr_leftseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_gsr_leftseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_ibi_leftseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(output_dir + "Processing/emp_ibi_leftseat_" + trial_settings.Scenario[this_folder] + ".csv already exists")
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Ibi.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Ibi.log"
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_ibi_leftseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_ibi_leftseat, "Processing/", "emp_ibi_leftseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_ibi_leftseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_temp_leftseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(
                output_dir + "Processing/emp_temp_leftseat_" + trial_settings.Scenario[this_folder] + ".csv already exists"
            )
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Temp.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device_Temp.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_temp_leftseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_temp_leftseat, "Processing/", "emp_temp_leftseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_temp_leftseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/smarteye_leftseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(
                output_dir + "Processing/smarteye_leftseat_" + trial_settings.Scenario[this_folder] + ".csv already exists"
            )
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (
                    helper.crew_dir
                    + "Synched/"
                    + str(trial_settings.RunDateTime[this_folder])
                    + "/SmartEye_Left_Seat_ET.log"
                )
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/SmartEye_Left_Seat_ET.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                smarteye_leftseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    smarteye_leftseat, "Processing/", "smarteye_leftseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: smarteye_leftseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/abm_rightseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(output_dir + "Processing/abm_rightseat_" + trial_settings.Scenario[this_folder] + ".csv already exists")
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: " + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/ABM-1.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/ABM-1.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                abm_rightseat = helper.adjust_timestamps(this_table)
                helper.store_file(abm_rightseat, "Processing/", "abm_rightseat_" + str(trial_settings.Scenario[this_folder]))
                print("converted to: abm_rightseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_acc_rightseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(
                output_dir + "Processing/emp_acc_rightseat_" + trial_settings.Scenario[this_folder] + ".csv already exists"
            )
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Acc.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Acc.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_acc_rightseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_acc_rightseat, "Processing/", "emp_acc_rightseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_acc_rightseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_bvp_rightseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(
                output_dir + "Processing/emp_bvp_rightseat_" + trial_settings.Scenario[this_folder] + ".csv already exists"
            )
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Bvp.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Bvp.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_bvp_rightseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_bvp_rightseat, "Processing/", "emp_bvp_rightseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_bvp_rightseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_gsr_rightseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(
                output_dir + "Processing/emp_gsr_rightseat_" + trial_settings.Scenario[this_folder] + ".csv already exists"
            )
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Gsr.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Gsr.log"
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_gsr_rightseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_gsr_rightseat, "Processing/", "emp_gsr_rightseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_gsr_rightseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_ibi_rightseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(
                output_dir + "Processing/emp_ibi_rightseat_" + trial_settings.Scenario[this_folder] + ".csv already exists"
            )
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Ibi.log")
            )
            try:
                # breakpoint()
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Ibi.log"
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_ibi_rightseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_ibi_rightseat, "Processing/", "emp_ibi_rightseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_ibi_rightseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/emp_temp_rightseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(
                output_dir + "Processing/emp_temp_rightseat_" + trial_settings.Scenario[this_folder] + ".csv already exists"
            )
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Temp.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/Emp_Emp_Device2_Temp.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                emp_temp_rightseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    emp_temp_rightseat, "Processing/", "emp_temp_rightseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: emp_temp_rightseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/smarteye_rightseat_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(
                output_dir + "Processing/smarteye_rightseat_" + trial_settings.Scenario[this_folder] + ".csv already exists"
            )
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (
                    helper.crew_dir
                    + "Synched/"
                    + str(trial_settings.RunDateTime[this_folder])
                    + "/SmartEye_Right_Seat_ET.log"
                )
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/SmartEye_Right_Seat_ET.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                smarteye_rightseat = helper.adjust_timestamps(this_table)
                helper.store_file(
                    smarteye_rightseat, "Processing/", "smarteye_rightseat_" + str(trial_settings.Scenario[this_folder])
                )
                print("converted to: smarteye_rightseat_" + trial_settings.Scenario[this_folder])

        if os.path.exists(output_dir + "Processing/ifd_cockpit_" + trial_settings.Scenario[this_folder] + ".csv"):
            print(output_dir + "Processing/ifd_cockpit_" + trial_settings.Scenario[this_folder] + ".csv already exists")
            continue
        else:
            this_table = pd.DataFrame()
            print(
                "processing: "
                + (helper.crew_dir + "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/IFD_COCKPIT.log")
            )
            try:
                this_table = helper.read_bucket_log(
                    "Synched/" + str(trial_settings.RunDateTime[this_folder]) + "/IFD_COCKPIT.log",
                )
            except:
                print("file doesnt exist")
            if not this_table.empty:
                ifd_cockpit = helper.adjust_timestamps(this_table)
                helper.store_file(ifd_cockpit, "Processing/", "ifd_cockpit_" + str(trial_settings.Scenario[this_folder]))
                print("converted to: ifd_cockpit_" + trial_settings.Scenario[this_folder])

        # if args.Push2Cloud:
        #     helper.sync_crew_folder_storage()
