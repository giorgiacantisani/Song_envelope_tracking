%% Generate CND files from data segmented and preprocessed in Python
%% Do bad channel rejection here as opposed to Python

%% Init
close all;
clear; clc;

% Add other directories to path
addpath ..\..\MATLAB\cnsp_utils
addpath ..\..\MATLAB\cnsp_utils\cnd
addpath ..\..\MATLAB\eeglab

%% Parameters preprocessing
dataEegFolder = '.\outputs\EEG\';
dataCNDFolder = '.\outputs\CND\';
pre = '1-8Hz';
reref_type = 'Matoids';
downfreq = 64;

%% General parameters
NTRIALS = 18;
conditions = {'melody', 'song', 'speech' };
subjects = {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16'};
dataType = 'EEG';
deviceName = 'Biosemi';
chanlocs = load([dataCNDFolder, 'chanlocs64.mat']).chanlocs;

%% Generate data stim for each condition
for condition = conditions
    cond = condition{1};

    % Load data stim for this condition
    load([dataCNDFolder, cond, '/dataStim.mat'], 'stim');

    % Iterate over subjects
    for idxSbj = 1:length(subjects)
        subject = subjects{idxSbj};
        disp(['Processing subject ', subject])
        curr_path = [dataEegFolder,pre,'\',reref_type,'\',subject,'\'];

        % Get preprocessing pipeline from csv
        T = readtable([curr_path, 'preprocessing_pipeline.csv']);
        preprocessingPipeline = {['LPF ', T.LPF, 'Hz'], ... 
                                 ['HPF ', T.HPF, 'Hz'], ...
                                 ['Reref ', T.reref_type], ...
                                 ['Down ', downfreq, 'Hz']};

        % Create data struct for this subject
        eeg = struct();
        eeg.dataType = dataType;
        eeg.deviceName = deviceName;
        eeg.condition = cond;
        eeg.chanlocs = chanlocs;
        eeg.extChan{1,1} = struct('description', 'Mastoids', 'chanlocs', []);
        eeg.fs = downfreq;
        eeg.reRef = T.reref_type;
        eeg.preprocessingPipeline = preprocessingPipeline;
        
        % Fill data structure with trials
        for idxTrial = 1: NTRIALS
            stim_name = stim.audioFiles{idxTrial};
            stim_path = [curr_path,stim_name,'.mat'];
            % Handle missing trials
            if ~exist(stim_path, 'file')
                eeg.nonemptyTrials{idxTrial} = 0; 
                eeg.data{idxTrial} = [];
                eeg.extChan{1,1}.data{idxTrial} = [];
                disp([stim_name, ' does not exist'])
            else
                eeg.nonemptyTrials{idxTrial} = 1;
                load(stim_path);
                eeg.data{idxTrial} = trial_data';
                eeg.extChan{1,1}.data{idxTrial} = trial_mastoids';
            end
        end

        % Interpolate bad channels
        disp('Interpolating bad channels...')
        if isfield(eeg,'chanlocs')
            for i = 1:numel(eeg.data)
                eeg.data{i} = removeBadChannels(eeg.data{i},eeg.chanlocs);
            end
            eeg = cndNewOp(eeg,'removeBadChannels');
        end

        % Saving preprocessed data
        eegFolder = [dataCNDFolder, cond, '\', pre, '\', reref_type];
        mkdir(eegFolder)
        eegPreFilename = [eegFolder, '\pre_dataSub', num2str(idxSbj), '.mat'];
        disp(['Saving preprocessed EEG data: ', cond, ' of subject ' num2str(idxSbj)])
        save(eegPreFilename,'eeg');

    end

end
disp('Done!')