# soteria-processing-python

## Setup Access and Software
- [ ] A number of steps need to be taken to get access to the data and the code that we used for transferring the data, initial quality assurance checks, and initial preprocessing and analyses.
 
- [ ] In order to use the code that is stored in this github repository, you will need to install Python 3 and several modules:
```
tensorflow
matplotlib
scipy
pandas
numpy
mne
boto3
```
- [ ] "pip install ."" from your local repository will install the version that was last used for development.

## Data Quality Assurance And Processing Steps
Description of the order of events from data synching to most recent analyses (and the associated script names):
 

### Data Transfer (Will not need repeated unless a problem is discovered)

- [ ] Data Synching/Collection - MAPPS synch done in real-time room while adaptors were running on respective acquisition machines
- [ ] Manual Transfer of raw and synched files from MAPPS machine (and acq. machines for raw) to external hard drive
- [ ] Auto Transfer of files from external drive to Box (transfer_to_box.m)
- [ ] Compare file size between external drive and Box to confirm complete transfer (compare_file_size.py)
- [ ] For public access, we eventually decided to copy the Box repository to another cloud based data repository (AWS S3) using AWS CLI commands.


All available scripts have a similar structure. Each script has a number of arguments, all of which have defaults, including a local processing folder (default set to your home directory). Other arguments include crews, scenarios, as well as various toggles (variable assigned to 0 or 1) for performing creating figures or analyses. There is also a "Push2Cloud" flag, that will only function for the admins of  the data repository.

An example of calling a script with all defaults except for the processing directory is as follows:
```
$ python3 export_raw_data -d "/home/$user/nasa_processing_dir/"
```

If wanting to only analyze a subset of the data, you can specify specific crews and/or scenarios as follows:
```
$ python3 export_raw_data -c '01','02','03' -s '1','2','5'  -d "/home/$user/nasa_processing_dir/"
```

Some notes regarding script arguments:
- [ ] The crews must be two digits, (0 before a single digit number)
- [ ] There is no crew 12 (see data paper for details)
- [ ] There is no scenario 4 (see data paper for details)
- [ ] If you specify a directory (-d) other than the default for export_raw_data, you will need to specify for every subsequent step


Scripts to execute in order of execution:
 
- [ ] Export data from MAPPS log files to csv (export_raw_data.py)

- [ ] Check whether the files exist, and generate heatmaps to visualize how many files missing (check_rawdata_exists.py and check_rawdata_exists_summary.py)

- [ ] Find the events the index of the "events" for each scenario and participant (find_events.py)

### Smarteye Preproccessing (preprocess_smarteye.py)  
- [ ] Preprocessing script that gathers a timeseries of epochs (average over the epoch) of multiple variables including gaze variance, gaze velocity, head rotation, and pupil diameter. An data point (frame) is only included if the gaze vector velocity is less than 700degrees/sec, and the gaze quality is greater than 60 %. Pupil diameter indices are included if the data quality was greater than 40%. (preprocess_smarteye.py)
This script includes additional flags for plot creations. 
- [ ] plot_qatable toggle if you want to plot the heatmap of available files (-p1)
- [ ] plot_aoi toggle if you want to see the scatter plot of the available data shown for each area of interest (aoi) (-p2)
- [ ] "time_per_epoch_4_analysis" is hard coded, and is the number of seconds per epoch that will be used to generate the analysis data structure. 

### EEG Preprocessing (preprocess_eeg.py)
- [ ] Preprocessing script that ultimately takes a matrix of power band frequencies for EEG and generates Task Load and Engagement indices. The raw EEG signals were first viewed and coded (good, ok, bad). This information is stored in an xlsx (eeg_electrode_quality_vector.xlsx). All signals were filtered before calculating the power spectrum and subsequently the indices. (preprocess_eeg.py)
This script includes additional flags for plot creations. 
- [ ] plot_raw toggle will generate figures that allows to view and scroll through the raw EEG signals (-p1)
- [ ] plot_individual toggle will generate power spectrum time series for each electrode (-p2)
- [ ] "number_of_epochs" is hard coded, and is the number of epochs to breakup the raw signal into

### EKG Preprocessing (preprocess_ekg.py)
- [ ] Preprocessing script that takes the raw ekg, runs a peak finding algorithm on it, and then calculates beats per minute and heart rate variability from this data. (preprocess_ekg.py)
This script includes additional flags for plot creations. 
- [ ] plot_qa_figs toggle will generate figures that allows to view and scroll through raw EKG signals (-p1)
	- [] start scrolling by clicking on scroll bar and the axes will auto zoom
- [ ] "time_per_epoch_4_analysis" is hard coded, and is the time in seconds for each epoch

### Plot Available
Each modality (Smarteye, EEG, and EKG) have a plot_availability script, that provides a high level view of the percent of data that is in good shape, lost, or could use additional preprocessing steps to improve. 

### Gather Results
There exists a gathering script for each data type. The purpose of this gathering script is to go through and concatenate all subject/crew results into a single dataframe, store these results in a csv, and create some figures of this data.

## Data
The data is publicly available at the Registry of Open Data on [AWS Registry](https://registry.opendata.aws/) and can be found by searching for "soteria", or more specifically, "nasa-soteria-data". The data can be pulled and listed using standard aws cli functions (see [AWS CLI](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-bucket-intro.html)section).

## Contributing
Our primary goal of releasing the code specific to a publicly available data set is to allow the public to help define and classify resilient behavior specifically in aviation operations and in general. The mechanisms for enabling this are the release of a rich, comprehensive data set, along with the code that can be used to organize and preprocess the data. 

Our hope is that the others will contribute back to both the data and the code repositories as significant improvements are made. 
Additions to the data set need to be approved by the administrator of the data set, therefore, please reach out to either Tyler Fettrow or Chad Stephens (contact information listed on title page).

## Authors and acknowledgment
The work was supported by NASA Aeronautics Research Mission Directorate Airspace Operations and Safety Program System-Wide Safety Project.

Chad Stephens, Lance Prinzel, Jon Holbrook,  Michael Stewart, Kathryn Ballard, and Daniel Kiggins designed the experiment.

This work supported from members of NASA Langley Research Center Simulation Development and Analysis Branch specifically Jim Barnes, Lon Kelly, Julie Timmons, Tom Feigh, Matthew Miser, and others. Alex Romero, air traffic control subject matter expert, served as the live air traffic controller interacting with flight crews during all simulation scenarios. Captain Brent Bushnell, commercial aviation subject matter expert, tested all simulated flight scenarios and advised NASA research team prior to the study being conducted.

Chad Stephens, Lance Prinzel, Jon Holbrook, Michael Stewart, Kathryn Ballard, Tyler Fettrow, Sepher Bastami and Daniel Kiggins carried out the experimental data collection.

Tyler Fettrow and Chad Stephens curated the dataset and produced the original example code in this repository. 

## Project status
There are currently several planned additions to the data set that will be uploaded at a future date.
First, the most critical is the addition of improved "event" identification. Currently the events are coded based on the timing of a particular call from ATC (i.e., deviate course), or a specific event occurring in the simulation (i.e., wake turbulence). However, it is possible that the ATC call timing differed between crews due to the ability of the crew to foresee the problem, or some other unforeseen circumstance. Regardless, there is a need for a more robust methodology for identifying the stressor events. Since each scenario had video and audio recordings, we plan to have observations completed by The LOSA Collaborative and American Airlines LIT that will provide resilience and workload classifications for each scenario and crew. This will provide expert level classification of events for improved future analyses. 

Additionally, we plan on having all the videos transcribed. We have attempted to use language processing software to transcribe, but due to the nuanced nature of air traffic communications, some language was misinterpreted. Therefore we plan on having experts check and correct the automated software results prior to releasing this data publicly. 

As features of data or code analyses are manufactured, we will provide updated details here.

## License
Notices:
Copyright 2024 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.
 
This software calls the following third-party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing. 
 
Disclaimers:
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.