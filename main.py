from AudioSeg import energy, windows, rising_edges, GetTime
import os
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import json

def extract_silence(input_file, output_dir, min_silence_length, silence_threshold, step_duration, dry_run=False):
    input_filename = input_file
    window_duration = min_silence_length
    if step_duration is None:
        step_duration = window_duration / 10.0
    else:
        step_duration = step_duration
    
    output_filename_prefix = os.path.splitext(os.path.basename(input_filename))[0]

    print("Splitting {} where energy is below {}% for longer than {}s.".format(
        input_filename,
        silence_threshold * 100.,
        window_duration
        )
    )

    sample_rate, samples = input_data = wavfile.read(filename=input_filename, mmap=True)

    max_amplitude = np.iinfo(samples.dtype).max
    # print(max_amplitude)

    max_energy = energy([max_amplitude])
    # print(max_energy)

    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)

    signal_windows = windows(
        signal=samples,
        window_size=window_size,
        step_size=step_size
    )

    window_energy = (energy(w) / max_energy for w in tqdm(
        signal_windows,
        total=int(len(samples) / float(step_size))
    ))

    window_silence = (e > silence_threshold for e in window_energy)

    cut_times = (r * step_duration for r in rising_edges(window_silence))

    print("Finding silences...")
    cut_samples = [int(t * sample_rate) for t in cut_times]
    cut_samples.append(-1)

    cut_ranges = [(i, cut_samples[i], cut_samples[i+1]) for i in range(len(cut_samples) - 1)]

    video_sub = {str(i) : [str(GetTime(((cut_samples[i])/sample_rate))), 
                           str(GetTime(((cut_samples[i+1])/sample_rate)))] 
                           for i in range(len(cut_samples) - 1)}
    
    for i, start, stop in tqdm(cut_ranges):
        output_file_path = "{}_{:03d}.wav".format(
            os.path.join(output_dir, output_filename_prefix),
            i
        )
        if not dry_run:
            print("Writing file {}".format(output_file_path))
            wavfile.write(
                filename=output_file_path,
                rate=sample_rate,
                data=samples[start:stop]
            )
        else:
            print("Not writing file {}".format(output_file_path))
            
    with open (os.path.join(output_dir, output_filename_prefix+'.json'), 'w') as output:
        json.dump(video_sub, output)
    
    return len(cut_ranges)


def multiple(input_dir, output_dir, min_silence_length, silence_threshold, step_duration, dry_run=False):
    filenames = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if i.endswith('.wav')]
    res = []
    for f in filenames:
        n = extract_silence(f, output_dir, min_silence_length, silence_threshold, step_duration, dry_run)
        fname = f.split('/')[-1]
        res.append((fname, n))
    return res

#0.8, 3e-4, 0.03/10 --> 81.545
#1.0, 5e-4, 1.0/10 --> 86.695
#1.0, 1.0, 5e-4, 0.5/10 --> 86.266
import compare
import pickle

# for i in range(1, 6):
#     for j in range(1, 10):
#         p1 = i * 1e-4
#         p2 = j * 1e-3
#         res = multiple('./data', './output', 1.0, p1, p2, True)
#         # with open('tmp.txt', 'w') as f:
#         #     print(res, file=f)
#         with open('./tmp.pkl', 'wb') as f:
#             pickle.dump(res, f)
#         c = compare.compare()
#         with open('./tmp.txt', 'a') as f:
#             print(p1, p2, c, file=f)

# res = multiple('./data', './output', 1.0, 1e-4, 0.005, True)
#         # with open('tmp.txt', 'w') as f:
#         #     print(res, file=f)
# with open('./tmp.pkl', 'wb') as f:
#     pickle.dump(res, f)
# c = compare.compare()
# print(c)
