from scipy.io import wavfile
import os
import numpy as np
import argparse
from tqdm import tqdm
import json

from datetime import datetime, timedelta

# Utility functions

def GetTime(video_seconds):

    if (video_seconds < 0) :
        return 00

    else:
        sec = timedelta(seconds=float(video_seconds))
        d = datetime(1,1,1) + sec

        instant = str(d.hour).zfill(2) + ':' + str(d.minute).zfill(2) + ':' + str(d.second).zfill(2) + str('.001')
    
        return instant

def GetTotalTime(video_seconds):

    sec = timedelta(seconds=float(video_seconds))
    d = datetime(1,1,1) + sec
    delta = str(d.hour) + ':' + str(d.minute) + ":" + str(d.second)
    
    return delta

def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]

def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))

# not used
def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1
