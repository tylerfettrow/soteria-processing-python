 function [dataoutput] = adjust_timestamps(datainput)
timestamps_uint = uint64(datainput.UserTimeStamp);
timestamps_datetime = datetime(double(timestamps_uint)/1e7,'ConvertFrom','epochtime','Epoch','1-Jan-0001','Format','dd-MMM-yyyy HH:mm:ss.SSSSSSSSS');
timestamps_time = seconds(timeofday(timestamps_datetime));
adjusted_time = timestamps_time - timestamps_time(1);
datainput{:,1} = adjusted_time;
dataoutput = datainput;
end