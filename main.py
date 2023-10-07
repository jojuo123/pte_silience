from AudioSeg import energy, windows, rising_edges, GetTime
import os
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import numpy as np

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
    
    return len(cut_ranges), cut_ranges, sample_rate


def multiple(input_dir, output_dir, min_silence_length, silence_threshold, step_duration, dry_run=False):
    filenames = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if i.endswith('.wav')]
    res = []
    for f in filenames:
        n = extract_silence(f, output_dir, min_silence_length, silence_threshold, step_duration, dry_run)
        fname = f.split('/')[-1]
        res.append((fname, n))
    return res

def handle_score_1_2(talk_range, short_talk_range=(1, 1.5, 2)):
    min_talk, med_talk, high_talk = short_talk_range
    lens = talk_range.size 
    if (lens == (talk_range <= min_talk).sum()):
        return 0
    first_range = talk_range >= min_talk & talk_range < med_talk
    if first_range.sum() >= 4:
        return 2
    else:
        second_range = talk_range >= med_talk < high_talk
        num_second_range = second_range.sum()
        if num_second_range >= 2:
            return 2
        else:
            high_range = talk_range >= high_talk
            if high_range.sum() >= 1:
                return 2
            else:
                return 1
    
def fluency_detecting(input_file, output_dir, short_silence_range=(0.3, 1), silence_threshold=1e-4, step_duration=0.005, dry_run=False):
    min_silence, max_silence = short_silence_range
    n, cut_ranges, sample_rate = extract_silence(input_file, output_dir, min_silence, silence_threshold, step_duration, dry_run)
    if n == 1:
        return 5
    score = -1
    talk_range = np.array([(e - s) / sample_rate for s, e in cut_ranges])
    starts = [s for s, _ in cut_ranges]
    starts = np.array(starts[1:])
    ends = [e for _, e in cut_ranges]
    ends = np.array(ends[:-1])
    silence_range = starts - ends
    short_silence = silence_range < max_silence
    count_short_silence = short_silence.sum()
    short_talk_range = (1, 1.5)
    if count_short_silence <=2:
        score = 3
    elif count_short_silence <= 9:
       score = handle_score_1_2(talk_range, short_talk_range)
    else:
        score = 0
    return score 
        

    
    
    
    # for i, (s, e) in enumerate(cut_ranges):
    #     if e - s >= 1.5:
    #         score = 2
    #     if i == 0:
    #         continue
    #     e_prev = cut_ranges[i-1][1]
    #     if s - e_prev > max_silence_range:
    #         continue
    #     else:
    #         if s - e <= 0.5:
    #             head[i] = head[i-1]
    #             dic[head[i]] += 1


#có track được 1 pattern?
#đọc giật đọc giật --> quy định sau --> pause dài 
# ngắn - pause - ngắn nếu khoảng cách giữa 2 pattern -> ngắn 
# số giây 
# tốc độ nói trung bình -> 180 wpm
# --> interval giữa 2 pause = 0.3-1s
# barem:
# mức 0: 
# đọc ngắn - pause lâu 
# mức 2: đọc 2 chữ liên tiếp trong 1 cụm 6 chữ --> quy về đọc 6 chữ liên tiếp:
# đk: pattern >=2s 
# 2 đoạn đọc được 1.5s 
# ko xuất hiện trait 0
#mức 3: xuất hiện tối đa 2 pattern đọc ngắn-nghỉ ngắn-đọc ngắn
# bị đến lần thứ 3 --> quay về xét điểm 1 2 
#  
# mức 3+: 
# 
#  
#0.8, 3e-4, 0.03/10 --> 81.545
#1.0, 5e-4, 1.0/10 --> 86.695
#1.0, 1.0, 5e-4, 0.5/10 --> 86.266
# import compare
# import pickle

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
