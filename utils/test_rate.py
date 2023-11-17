

from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
from main import extract_silence
from derive import derivative, derivative2

# samplerate, data = wavfile.read('./data/RA268_2023-06-18_17_13_16-1994.wav')
# samplerate2, data2 = wavfile.read('./data/RA268_2023-06-18_17_57_07-5e87.wav')
# # length = data.shape[0] / samplerate
# length = 5
# import matplotlib.pyplot as plt
# import numpy as np
# time = np.linspace(0., length, data.shape[0])
# time2 = np.linspace(0., length, data2.shape[0])
# print(data.shape)
# plt.plot(time, data[:,], label="Left channel")
# plt.plot(time2, data2[:, ], label="Right channel")
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()
samplerate, data = wavfile.read('./data/RA268_2023-06-18_17_13_16-1994.wav')
samplerate, data = wavfile.read('./data/RA268_2023-06-18_17_57_07-5e87.wav')
_, talk_range, _ = extract_silence(None, None, 0.4, 3e-4, 0.02, sample_rate=samplerate, samples=data)
skeleton = derivative2(data, samplerate, talk_range)
print(skeleton)