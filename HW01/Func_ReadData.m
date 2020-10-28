function [Output,frequency,mean_rate] = Func_ReadData (str)
    CurrentFolder = pwd;
    path = [CurrentFolder,'\Data\Spike_and_Log_Files\',str];
    file_list = dir(path);
    events_val = cell(0);
    hdr_val = cell(0);
    check = 0;
    for i = 1 : length(file_list)
        if ((~isempty(strfind(file_list(i).name,'msq1d.sa0')) || ~isempty(strfind(file_list(i).name,'msq1D.sa0'))) && isempty(strfind(file_list(i).name,'.sa0.')))
            if(~check)
                counter = 1;
                events_val{1} = (fget_spk([path,'\',file_list(i).name]))';
                hdr_val{1} = fget_hdr([path,'\',file_list(i).name]);
                check = 1;
                logfile=importdata([path,'\',file_list(i-1).name]);
                logfile=logfile.textdata;
                c=logfile{13};
                frequency=str2double(c(31:end));
                rate = length(events_val{1})/(32767/frequency);
            else
                counter = counter + 1;
                events_val = [events_val,(fget_spk([path,'\',file_list(i).name]))'];
                rate = rate + length(events_val{counter})/(32767/frequency);
                hdr_val = [hdr_val,fget_hdr([path,'\',file_list(i).name])];
            end
        end
    end
    Output = struct('events',events_val,'hdr',hdr_val);
    mean_rate = rate/counter;
    fclose('all');
end