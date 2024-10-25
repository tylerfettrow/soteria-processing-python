import numpy as np
import math
from numpy import linalg as la
import os
import pandas as pd
import subprocess
from datetime import datetime
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import logging
import sys

for name in logging.Logger.manager.loggerDict.keys():
    if (
        ("boto" in name)
        or ("urllib3" in name)
        or ("s3transfer" in name)
        or ("boto3" in name)
        or ("botocore" in name)
        or ("nose" in name)
    ):
        logging.getLogger(name).setLevel(logging.CRITICAL)


class HELP:
    # global local_process_dir
    # local_process_dir = os.path.expanduser('~')
    def __init__(self):
        self.crew_dir = ""
        self.local_process_dir = os.path.expanduser("~") + "/nasa-soteria-data/"
        self.s3c = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        self.s3_bucket_name = "nasa-soteria-data"

    def read_local_table(self, file_name):
        # obj = self.s3c.get_object(Bucket= self.s3_bucket_name, Key= self.crew_dir + file_name)
        data = pd.read_table(self.local_process_dir + self.crew_dir + file_name, delimiter=",")
        return data

    def read_bucket_table(self, file_name):
        obj = self.s3c.get_object(Bucket=self.s3_bucket_name, Key=self.crew_dir + file_name)
        data = pd.read_table(obj["Body"], delimiter=",")
        return data

    def read_bucket_log(self, file_name):
        obj = self.s3c.get_object(Bucket=self.s3_bucket_name, Key=self.crew_dir + file_name)
        data = pd.read_table(obj["Body"], delimiter="\t")
        return data

    def check_bucket_log(self, file_name):
        obj = self.s3c.get_object(Bucket=self.s3_bucket_name, Key=self.crew_dir + file_name)
        # data = pd.read_table(obj['Body'],delimiter="\t")
        return obj

    def read_bucket_xlsx(self, file_name, sub_worksheet):
        if not os.path.isfile(self.local_process_dir+file_name):
            cmd = "aws s3 cp s3://nasa-soteria-data/{} {}".format(file_name, self.local_process_dir+"Analysis/")
            output_dir = self.local_process_dir+"Analysis/"
            os.makedirs(output_dir, exist_ok=True)
            subprocess.call(cmd, shell=True)
        data = pd.read_excel(self.local_process_dir+file_name, sub_worksheet)
        return data

    def store_file(self, dataframe, folder_name, file_name):
        output_dir = self.local_process_dir + self.crew_dir + folder_name
        os.makedirs(output_dir, exist_ok=True)
        dataframe.to_csv(
            output_dir + file_name + ".csv",
            index=False,
        )

    def ParseCrewAndScenario(self, sys_argv, crew_arg, scenario_arg):
        crew_arr = crew_arg
        scenarios = scenario_arg
        if len(sys.argv) > 1:
            if sys.argv[1] == "-c":
                crew_arr = sys.argv[2].split(",")
            if sys.argv[1] == "-s":
                scenarios = sys.argv[2].split(",")
            if len(sys.argv) > 3:
                if sys.argv[3] == "-c":
                    crew_arr = sys.argv[4].split(",")
                if sys.argv[3] == "-s":
                    scenarios = sys.argv[4].split(",")
        return crew_arr, scenarios

    def sync_crew_folder_storage(self):
        print("synching " + self.crew_dir)
        # breakpoint()
        cmd = "aws s3 sync {}{}/Figures/ s3://nasa-soteria-data/{}/Figures".format(
            self.local_process_dir, self.crew_dir, self.crew_dir
        )
        print(cmd)
        subprocess.call(cmd, shell=True)

        cmd = "aws s3 sync {}{}/Processing/ s3://nasa-soteria-data/{}/Processing".format(
            self.local_process_dir, self.crew_dir, self.crew_dir
        )
        print(cmd)
        subprocess.call(cmd, shell=True)

        cmd = "aws s3 sync {}{}/Analysis/ s3://nasa-soteria-data/{}/Analysis".format(
            self.local_process_dir, self.crew_dir, self.crew_dir
        )
        print(cmd)
        subprocess.call(cmd, shell=True)

    def getSubWorksheet(self, seat):
        """
        grabbing the naming scheme for the electrode worksheet
        """
        b = []
        if (self.crew_dir == "Crew_01/") & (seat == "abm_leftseat"):
            b = "Crew_01_Left"
        elif (self.crew_dir == "Crew_01/") & (seat == "abm_rightseat"):
            b = "Crew_01_Right"
        elif (self.crew_dir == "Crew_02/") & (seat == "abm_leftseat"):
            b = "Crew_02_Left"
        elif (self.crew_dir == "Crew_02/") & (seat == "abm_rightseat"):
            b = "Crew_02_Right"
        elif (self.crew_dir == "Crew_03/") & (seat == "abm_leftseat"):
            b = "Crew_03_Left"
        elif (self.crew_dir == "Crew_03/") & (seat == "abm_rightseat"):
            b = "Crew_03_Right"
        elif (self.crew_dir == "Crew_04/") & (seat == "abm_leftseat"):
            b = "Crew_04_Left"
        elif (self.crew_dir == "Crew_04/") & (seat == "abm_rightseat"):
            b = "Crew_04_Right"
        elif (self.crew_dir == "Crew_05/") & (seat == "abm_leftseat"):
            b = "Crew_05_Left"
        elif (self.crew_dir == "Crew_05/") & (seat == "abm_rightseat"):
            b = "Crew_05_Right"
        elif (self.crew_dir == "Crew_06/") & (seat == "abm_leftseat"):
            b = "Crew_06_Left"
        elif (self.crew_dir == "Crew_06/") & (seat == "abm_rightseat"):
            b = "Crew_06_Right"
        elif (self.crew_dir == "Crew_07/") & (seat == "abm_leftseat"):
            b = "Crew_07_Left"
        elif (self.crew_dir == "Crew_07/") & (seat == "abm_rightseat"):
            b = "Crew_07_Right"
        elif (self.crew_dir == "Crew_08/") & (seat == "abm_leftseat"):
            b = "Crew_08_Left"
        elif (self.crew_dir == "Crew_08/") & (seat == "abm_rightseat"):
            b = "Crew_08_Right"
        elif (self.crew_dir == "Crew_09/") & (seat == "abm_leftseat"):
            b = "Crew_09_Left"
        elif (self.crew_dir == "Crew_09/") & (seat == "abm_rightseat"):
            b = "Crew_09_Right"
        elif (self.crew_dir == "Crew_10/") & (seat == "abm_leftseat"):
            b = "Crew_10_Left"
        elif (self.crew_dir == "Crew_10/") & (seat == "abm_rightseat"):
            b = "Crew_10_Right"
        elif (self.crew_dir == "Crew_11/") & (seat == "abm_leftseat"):
            b = "Crew_11_Left"
        elif (self.crew_dir == "Crew_11/") & (seat == "abm_rightseat"):
            b = "Crew_11_Right"
        elif (self.crew_dir == "Crew_12/") & (seat == "abm_leftseat"):
            b = "Crew_12_Left"
        elif (self.crew_dir == "Crew_12/") & (seat == "abm_rightseat"):
            b = "Crew_12_Right"
        elif (self.crew_dir == "Crew_13/") & (seat == "abm_leftseat"):
            b = "Crew_13_Left"
        elif (self.crew_dir == "Crew_13/") & (seat == "abm_rightseat"):
            b = "Crew_13_Right"
        return b

    def get_unique(self, list1):
        """get unique values from list"""
        # initialize a null list
        unique_list = []

        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        return unique_list

    def getCrewInt(self):
        """get the int value that corresponds to the crew number (for indexing through preallocated matrices)"""
        b = []
        if self.crew_dir == "Crew_01/":
            b = 1
        elif self.crew_dir == "Crew_02/":
            b = 2
        elif self.crew_dir == "Crew_03/":
            b = 3
        elif self.crew_dir == "Crew_04/":
            b = 4
        elif self.crew_dir == "Crew_05/":
            b = 5
        elif self.crew_dir == "Crew_06/":
            b = 6
        elif self.crew_dir == "Crew_07/":
            b = 7
        elif self.crew_dir == "Crew_08/":
            b = 8
        elif self.crew_dir == "Crew_09/":
            b = 9
        elif self.crew_dir == "Crew_10/":
            b = 10
        elif self.crew_dir == "Crew_11/":
            b = 11
        elif self.crew_dir == "Crew_12/":
            b = 12
        elif self.crew_dir == "Crew_13/":
            b = 13
        return b

    def sphere_stereograph(self, p):
        """project unit vector onto planar surfrace (@ref)"""
        [m, n] = p.shape

        s = np.divide(2.0, (1.0 + p[m - 1, 0:n]))
        ss = np.matlib.repmat(s, m, 1)

        f = np.zeros((m, 1))
        f[m - 1] = -1.0
        ff = np.matlib.repmat(f, 1, n)

        q = np.multiply(ss, p) + np.multiply((1.0 - ss), ff)

        b = q[0:2, :]

        return b

    def angle_diff(self, time_vector, direction_gaze):
        """
        WARNING: this could probably be made more efficient.
        Calculatees the degrees_per_sec between each frame of gaze vector
        """

        degree_per_sec_vector = np.zeros(direction_gaze.shape[1])

        time_diff = np.diff(time_vector)
        for this_frame in range(direction_gaze.shape[1] - 1):
            if time_diff[this_frame] != 0.0:
                degree_diff_vector = math.degrees(
                    2
                    * math.atan(
                        la.norm(
                            np.multiply(
                                direction_gaze[:, this_frame],
                                la.norm(direction_gaze[:, this_frame + 1]),
                            )
                            - np.multiply(
                                la.norm(direction_gaze[:, this_frame]),
                                direction_gaze[:, this_frame + 1],
                            )
                        )
                        / la.norm(
                            np.multiply(
                                direction_gaze[:, this_frame],
                                la.norm(direction_gaze[:, this_frame + 1]),
                            )
                            + np.multiply(
                                la.norm(direction_gaze[:, this_frame]),
                                direction_gaze[:, this_frame + 1],
                            )
                        )
                    )
                )
                degree_per_sec_vector[this_frame + 1] = degree_diff_vector / time_diff[this_frame]
            else:
                degree_per_sec_vector[this_frame + 1] = 0
        return degree_per_sec_vector

    def mean(self, a, n):
        """
        mean by specific dimension
        """
        sum = 0
        for i in range(n):
            for j in range(n):
                sum += a[i][j]
        return sum / (n * n)

    def variance(self, a, n, m):
        """Function for calculating variance"""
        sum = 0
        for i in range(n):
            for j in range(n):
                # subtracting mean
                # from elements
                a[i][j] -= m

                # a[i][j] = fabs(a[i][j]);
                # squaring each terms
                a[i][j] *= a[i][j]

        # taking sum
        for i in range(n):
            for j in range(n):
                sum += a[i][j]

        return sum / (n * n)

    def adjust_timestamps(self, datainput):
        timestamps_time = np.zeros(len(datainput.UserTimeStamp))
        for this_index in range(datainput.UserTimeStamp.shape[0]):
            this_timestamp = datainput.UserTimeStamp[this_index] / 1e7
            this_timestamp.astype("int64")
            this_timestamp_datetime = datetime.fromtimestamp(this_timestamp)
            this_timestamp_time_string = str(this_timestamp_datetime.time())

            ## H:M:S -> seconds
            this_timestamp_time_string_split = this_timestamp_time_string.split(":")
            timestamps_time[this_index] = (
                float(this_timestamp_time_string_split[0]) * 3600
                + float(this_timestamp_time_string_split[1]) * 60
                + float(this_timestamp_time_string_split[2])
            )

        timestamps_time_adjusted = timestamps_time - timestamps_time[0]
        datainput.UserTimeStamp = timestamps_time_adjusted
        return datainput
