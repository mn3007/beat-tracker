#!/usr/bin/env python
# coding: utf-8

# In[151]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os, sys
import numpy as np
import librosa

def beatTracker(inputFile):

    # Import audio file
    x, Fs = librosa.load(inputFile)
    x_duration = len(x)/Fs
    
    # ONSET DETECTION USING SPECTRAL FLUX
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_NoveltySpectral.html
    #P.121
    # Window length and hop size
    N = 1024
    H = 512
    gamma = 100
    
    #p.310
    # Calculate the discrete STFT
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')

    # Apply logarithmic compression
    Y = np.log(1 + gamma * np.abs(X))
    # Calculate the nth discrete derivative
    Y_diff = np.diff(Y, n=1)
    # Only keeping the positive part (half-wave rectification)
    Y_diff[Y_diff < 0] = 0
    # Add up the positive differences over the frequency axis (accumulation step)
    onset_detection_function = np.sum(Y_diff, axis=0)
    onset_detection_function = np.concatenate((onset_detection_function, np.array([0])))
    Fs_onset_detection_function = Fs/H

    # Subtracting Local Average
    L = len(onset_detection_function)
    M_sec = 0.1
    M = int(np.ceil(M_sec * Fs_onset_detection_function))            
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(onset_detection_function[a:b])

    # Enhanced novelty function
    onset_detection_function =  onset_detection_function - local_average
    # Half-wave rectification
    onset_detection_function[onset_detection_function<0]=0
    # Normalize the novelty function
    onset_detection_function = onset_detection_function / max(onset_detection_function)        
            
    # GLOBAL TEMPO ESTIMATION
    tempo = librosa.beat.tempo(onset_envelope=onset_detection_function, sr=Fs)
    frames_per_beat = (Fs_onset_detection_function * 60) / tempo
    
    # BEAT TRACKING BY DYNAMIC PROGRAMMING
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S3_BeatTracking.html
    #p.335
        
    # Compute penalty function used for beat tracking
    penalty_frames = len(onset_detection_function)
    t = np.arange(1, penalty_frames) / frames_per_beat
    # Penalty function
    penalty = -np.square(np.log2(t))
    t = np.concatenate((np.array([0]), t))
    penalty = np.concatenate((np.array([0]), penalty))

    # Compute beat sequence using dynamic programming
    N = len(onset_detection_function)
    onset_detection_function = np.concatenate((np.array([0]), onset_detection_function))
    # Concatenation of '0' because of Python indexing conventions
    accumulated_score = np.zeros(N+1)
    maximisation_information = np.zeros(N+1, dtype=int)
    # Accumulated score
    accumulated_score[1] = onset_detection_function[1]
    maximisation_information[1] = 0

    # Forward calculation
    for n in range(2, N+1):
        m_indices = np.arange(1, n)
        # Deduct penalty function from onset detection function
        scores = accumulated_score[m_indices] + penalty[n-m_indices]
        # Find maximum
        maximum = np.max(scores)
        if maximum <= 0:
            accumulated_score[n] = onset_detection_function[n]
            maximisation_information[n] = 0
        else:
            accumulated_score[n] = onset_detection_function[n] + maximum
            maximisation_information[n] = np.argmax(scores) + 1

    # Backtracking
    beat_sequence_frames = np.zeros(N, dtype=int)
    k = 0
    beat_sequence_frames[k] = np.argmax(accumulated_score)
    while(maximisation_information[beat_sequence_frames[k]] != 0):
        k = k+1
        beat_sequence_frames[k] = maximisation_information[beat_sequence_frames[k-1]]
    beat_sequence_frames = beat_sequence_frames[0:k+1]
    beat_sequence_frames = beat_sequence_frames[::-1]
    beat_sequence_frames = beat_sequence_frames - 1
    
    # Convert from frames to time
    beat_sequence_time = librosa.frames_to_time(beat_sequence_frames, sr=Fs)
    
    return beat_sequence_time


# In[153]:


# Evaluation
get_ipython().run_line_magic('matplotlib', 'inline')
import mir_eval

def beatEval(reference_beats, estimated_beats):
    
    reference_beats = mir_eval.io.load_events('/Users/venusexmachina/Google Drive/MSc in Sound and Music Computing/ECS7006 Music Informatics/annotations/Albums-Cafe_Paradiso-05.txt')
    estimated_beats = mir_eval.io.load_events('/Users/venusexmachina/Google Drive/MSc in Sound and Music Computing/ECS7006 Music Informatics/estimated.txt')
    scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)
    
    # Plot the estimated and reference beats together

    
    # Evaluate        
    return scores


# In[ ]:




