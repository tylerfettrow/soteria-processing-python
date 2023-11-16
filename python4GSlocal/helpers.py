import numpy as np
import math
from numpy import linalg as la
import os
import shutil
from os.path import exists
import time
import pandas as pd
from google.cloud import storage

class HELP:
    def getElectrodeVectorWorksheet(self, crewID, seat):
        """ 
        grabbing the naming scheme for the electrode worksheet    
        """
        if ((crewID == 'Crew_01') & (seat == 'abm_leftseat')):
            b = 'Crew_01_Left'
        elif ((crewID == 'Crew_01') & (seat == 'abm_rightseat')):
            b = 'Crew_01_Right'
        elif ((crewID == 'Crew_02') & (seat == 'abm_leftseat')):
            b = 'Crew_02_Left'
        elif ((crewID == 'Crew_02') & (seat == 'abm_rightseat')):
            b = 'Crew_02_Right'
        elif ((crewID == 'Crew_03') & (seat == 'abm_leftseat')):
            b = 'Crew_03_Left'
        elif ((crewID == 'Crew_03') & (seat == 'abm_rightseat')):
            b = 'Crew_03_Right'
        elif ((crewID == 'Crew_04') & (seat == 'abm_leftseat')):
            b = 'Crew_04_Left'
        elif ((crewID == 'Crew_04') & (seat == 'abm_rightseat')):
            b = 'Crew_04_Right'
        elif ((crewID == 'Crew_05') & (seat == 'abm_leftseat')):
            b = 'Crew_05_Left'
        elif ((crewID == 'Crew_05') & (seat == 'abm_rightseat')):
            b = 'Crew_05_Right'
        elif ((crewID == 'Crew_06') & (seat == 'abm_leftseat')):
            b = 'Crew_06_Left'
        elif ((crewID == 'Crew_06') & (seat == 'abm_rightseat')):
            b = 'Crew_06_Right'
        elif ((crewID == 'Crew_07') & (seat == 'abm_leftseat')):
            b = 'Crew_07_Left'
        elif ((crewID == 'Crew_07') & (seat == 'abm_rightseat')):
            b = 'Crew_07_Right'
        elif ((crewID == 'Crew_08') & (seat == 'abm_leftseat')):
            b = 'Crew_08_Left'
        elif ((crewID == 'Crew_08') & (seat == 'abm_rightseat')):
            b = 'Crew_08_Right'
        elif ((crewID == 'Crew_09') & (seat == 'abm_leftseat')):
            b = 'Crew_09_Left'
        elif ((crewID == 'Crew_09') & (seat == 'abm_rightseat')):
            b = 'Crew_09_Right'
        elif ((crewID == 'Crew_10') & (seat == 'abm_leftseat')):
            b = 'Crew_10_Left'
        elif ((crewID == 'Crew_10') & (seat == 'abm_rightseat')):
            b = 'Crew_10_Right'
        elif ((crewID == 'Crew_11') & (seat == 'abm_leftseat')):
            b = 'Crew_11_Left'
        elif ((crewID == 'Crew_11') & (seat == 'abm_rightseat')):
            b = 'Crew_11_Right'
        elif ((crewID == 'Crew_12') & (seat == 'abm_leftseat')):
            b = 'Crew_12_Left'
        elif ((crewID == 'Crew_12') & (seat == 'abm_rightseat')):
            b = 'Crew_12_Right'
        elif ((crewID == 'Crew_13') & (seat == 'abm_leftseat')):
            b = 'Crew_13_Left'
        elif ((crewID == 'Crew_13') & (seat == 'abm_rightseat')):
            b = 'Crew_13_Right'
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


    def getCrewInt(self, crewID):
        """get the int value that corresponds to the crew number (for indexing through preallocated matrices)"""
        if crewID == "Crew_01":
            b = 1
        elif crewID == "Crew_02":
            b = 2
        elif crewID == "Crew_03":
            b = 3
        elif crewID == "Crew_04":
            b = 4
        elif crewID == "Crew_05":
            b = 5
        elif crewID == "Crew_06":
            b = 6
        elif crewID == "Crew_07":
            b = 7
        elif crewID == "Crew_08":
            b = 8
        elif crewID == "Crew_09":
            b = 9
        elif crewID == "Crew_10":
            b = 10
        elif crewID == "Crew_11":
            b = 11
        elif crewID == "Crew_12":
            b = 12
        elif crewID == "Crew_13":
            b = 13
        return b


    def sphere_stereograph(self, p):
        """ project unit vector onto planar surfrace (@ref) """
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
            if (time_diff[this_frame] != 0.0):
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
                degree_per_sec_vector[this_frame + 1] = (
                    degree_diff_vector / time_diff[this_frame]
                )
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

    def reset_folder_storage(self):
        """ 
        remove and create local folders that are used to store and subsequently push data/files to the repo
        """
        if exists("Figures"):
            shutil.rmtree("Figures", ignore_errors=True)
            time.sleep(3)
            os.mkdir("Figures")
        else:
            os.mkdir("Figures")
        if exists("Processing"):
            shutil.rmtree("Processing", ignore_errors=True)
            time.sleep(3)
            os.mkdir("Processing")
        else:
            os.mkdir("Processing")
        if exists("Analysis"):
            shutil.rmtree("Analysis", ignore_errors=True)
            time.sleep(3)
            os.mkdir("Analysis")
        else:
            os.mkdir("Analysis")

    def sync_crew_folder_storage(self):
        subprocess.call(
            'gsutil -m rsync -r Figures/ "gs://soteria_study_data/"'
            + crews_to_process[i_crew]
            + '"/Figures"',
            shell=True,
        )
        subprocess.call(
            'gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'
            + crews_to_process[i_crew]
            + '"/Processing"',
            shell=True,
        )
        subprocess.call(
            'gsutil -m rsync -r Analysis/ "gs://soteria_study_data/"'
            + crews_to_process[i_crew]
            + '"/Figures"',
            shell=True,
        )

    def getBucket(self):
        storage_client = storage.Client(project="soteria-fa59")
        bucket = storage.Bucket(
            storage_client, "soteria_study_data", user_project="soteria-fa59"
        )
        return bucket

    def read_bucket_table(self, process_dir_name, file_name):
        data = pd.read_table(
        ("gs://soteria_study_data/" + process_dir_name + file_name),
        delimiter=",",
        )
        return data