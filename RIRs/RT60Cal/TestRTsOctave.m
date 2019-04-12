% Qingju Liu, University of Surrey
% This programme calculates the mean RT60 time

p = mfilename('fullpath'); %gives the fullname
fileDirectory = fileparts(p); %gives the directory of the current running m-file
cd(fileDirectory);
clear all;close all;clc

% add directories to the data and called functions
% dataPath = '/blabalba';
% addpath(genpath(dataPath));

load B_format_RIRs_12BB01_Alfredo_S3A_16k.mat %rirs 36 * 4 * 20001 L*50*24

Results = zeros(5,36);
Use = true(1,36);
syn = 600;
for pos = 1:36,
    
    x = squeeze(rirs(pos,1,:));
    try
        rt60 = RTsOctave(x(syn:end),'graph',false,'method',1,'spec','full');
        %         % too check if this is affected by microphone delays
        %         rt60 = RTsOctave(x,'graph',true,'method',1,'spec','full');
        %         rt602 = RTsOctave(x(syn:end),'graph',true,'method',1,'spec','full');
        Results(:, pos) = rt60;
    catch
        Use(1,pos) = false;
        fprintf(['(',num2str(mic),',',num2str(pos),')']);
    end
    
end

RT60 = mean(Results(:))






