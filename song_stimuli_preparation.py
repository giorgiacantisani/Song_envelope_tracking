# Prepare stimuli for the experiment: postprocessing, normalization and add triggers
# Created by Giorgia Cantisani 02/06/2022

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import essentia.standard as es
from utils.melody.my_melosynth import *
from scipy.signal import hilbert, butter, lfilter, medfilt
from pedalboard import Reverb


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Paths
path_song = '../../Datasets/Dataset_song/originals/song/'
path_speech = '../../Datasets/Dataset_song/originals/speech/'
path_humm = '../../Datasets/Dataset_song/intermediates/melody/'
output_path = '../../Datasets/Dataset_song/intermediates/stimuli/' 
os.makedirs(output_path, exist_ok=True)

# Param
sampleRate=44100     # Hz
# Triggers
T_trigger = 0.010    # sec
A_trigger = 0.8
N = int(sampleRate*T_trigger)
# RMS-based filtering
amp_th = 0.0001
order = 6
cutoff = 100 
# Remove glitches
bandpassfiltering = False
max_freq_voice = 2000   # Hz --> C2
min_freq_voice = 100     # Hz --> C7
# Reverberation
add_reverb = False
reverb = Reverb() 
reverb.wet_level = 1.0
print(reverb)
rate = 1.2


# Find global max
energy_max = 0
for filename in librosa.util.find_files(path_humm, ext='wav'):
    file_name = filename.split('/')[-1]
    print('Processing audio file ' + file_name)

    speech = es.MonoLoader(filename=os.path.join(path_speech, file_name))().astype(np.float32)
    song = es.MonoLoader(filename=os.path.join(path_song, file_name))().astype(np.float32)
    melody = es.MonoLoader(filename=os.path.join(path_humm, file_name))().astype(np.float32)

    # Normalize energy: al has unitary energy
    song = song/float(np.std(song))
    melody = melody/float(np.std(melody))
    speech = speech/float(np.std(speech))

    candidate_max = max(float(np.max(melody)), float(np.max(song)), float(np.max(speech)))
    if candidate_max > energy_max:
        energy_max = candidate_max

# Normalize with global max
energy_max = float(energy_max)
for filename in librosa.util.find_files(path_song, ext='wav'):
    file_name = filename.split('/')[-1]
    file_idx = file_name.split('.')[0]
    print('Processing audio file ' + file_name)

    speech = es.MonoLoader(filename=os.path.join(path_speech, file_name))().astype(np.float32)
    song = es.MonoLoader(filename=os.path.join(path_song, file_name))().astype(np.float32)
    melody = es.MonoLoader(filename=os.path.join(path_humm, file_name))().astype(np.float32)

    if bandpassfiltering:
        song = butter_lowpass_filter(song, max_freq_voice, sampleRate, order)
        speech = butter_lowpass_filter(speech, max_freq_voice, sampleRate, order)

    # Denoise a bit the speech and song signals
    amplitudes = butter_lowpass_filter(np.abs(hilbert(song)), cutoff, sampleRate, order)
    song[amplitudes < amp_th] = 0
    amplitudes = butter_lowpass_filter(np.abs(hilbert(speech)), cutoff, sampleRate, order)
    speech[amplitudes < amp_th] = 0

    # Add reverb
    if add_reverb:
        melody = reverb.process(melody.astype(np.float32))

    song = librosa.effects.time_stretch(song, rate=rate)
    melody = librosa.effects.time_stretch(melody, rate=rate)

    # Normalize energy: al has unitary energy
    song = song/float(np.std(song))
    melody = melody/float(np.std(melody))
    speech = speech/float(np.std(speech))

    # Normalize over global energy
    song *= 0.8 / energy_max
    melody *= 0.8 / energy_max
    speech *= 0.8 / energy_max

    # Create triggers
    trigger_song = np.zeros(len(song))
    trigger_song[:N] = A_trigger
    trigger_melody = np.zeros(len(melody))
    trigger_melody[:N] = A_trigger
    trigger_speech = np.zeros(len(speech))
    trigger_speech[:N] = A_trigger

    # # Save new files
    # es.MonoWriter(filename=output_path + file_idx + '_song.wav', format='wav')(song)
    # es.MonoWriter(filename=output_path + file_idx + '_melody.wav', format='wav')(melody)
    # es.MonoWriter(filename=output_path + file_idx + '_speech.wav', format='wav')(speech)
    es.AudioWriter(filename=output_path + file_idx + '_song.wav', format='wav')(es.StereoMuxer()(song, trigger_song))  
    es.AudioWriter(filename=output_path + file_idx + '_melody.wav', format='wav')(es.StereoMuxer()(melody, trigger_melody))  
    es.AudioWriter(filename=output_path + file_idx + '_speech.wav', format='wav')(es.StereoMuxer()(speech, trigger_speech))  
print('Done!')