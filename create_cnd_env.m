%% Init
close all;
clear; clc;

% Set main paths
wav_path = "..\..\Datasets\Dataset_song\EEG_recordings\stimuli\";
outputFolder = '.\outputs\CND\';

% Add other directories to path
addpath ..\..\MATLAB\cnsp_utils
addpath ..\..\MATLAB\cnsp_utils\cnd

%% General parameters
fs_audio = 44100;
fs_down = 64;
NTRIALS = 18;
conditions = {'melody', 'song', 'speech' }; 
feature_names = {'envelope'};
additionalDetails = {'Energy'};

%% Generate data stim for each condition
for condition = conditions
    cond = condition{1};
    stim = struct();
    stim.fs = fs_down;
    stim.condition = cond;

    % Iterate over features
    for idx_feat = 1:length(feature_names)
        feature_name = feature_names{idx_feat};
        
        % Save metadata
        stim.names{idx_feat} = feature_name;
        stim.additionalDetails{idx_feat} = additionalDetails{idx_feat};

        % Iterate over stimuli
        path = strcat(wav_path,'\','*_',cond,'.wav');
        stim_file_names = dir(path);
        for idx_stim = 1:length(stim_file_names)
            stim_name = stim_file_names(idx_stim).name;

            % Retrive corresponding audio file
            C = strsplit(stim_name, '.');
            stim.audioFiles{idx_stim} = C{1};

            % Load auidio file
            wav_name = strcat(wav_path, C{1}, ".wav");
            [wav,fs_audio] = audioread(wav_name);
            
            % Remove click
            audio = wav(:, 1);
            
            % Compute envelope
            E = mTRFenvelope(audio,fs_audio,fs_down,1,1);
            E = E/max(abs(E)); 
            stim.data{idx_feat, idx_stim} = E;
        end
    end

    % Save data
    outputFilename = [outputFolder, cond, '/dataStim.mat'];
    disp(['Saving data stim for condition ', cond])
    save(outputFilename,'stim')
end
 
