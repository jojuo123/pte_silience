from AudioSeg import energy, windows
import os
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import json
from derive import derivative
import numpy as np
import pandas as pd
import syllable
import pydub
import timeit

def read_mp3(f):
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    return a.frame_rate, y


def extract_silence(min_silence_length, silence_threshold, step_duration, sample_rate, samples):
    window_duration = min_silence_length
    if step_duration is None:
        step_duration = window_duration / 10.0
    else:
        step_duration = step_duration

    wav_length = samples.shape[0]

    max_amplitude = np.iinfo(samples.dtype).max
    max_energy = energy([max_amplitude])
    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)

    signal_windows = windows(
        signal=samples,
        window_size=window_size,
        step_size=step_size
    )

    window_energy = (energy(w) / max_energy for w in signal_windows)

    window_silence = (e > silence_threshold for e in window_energy)

    window_silence = list(window_silence)
    talk_range = []
    prev = False
    s = 0
    e = 0
    for i in range(len(window_silence)):
        if window_silence[i] and not prev:
            s = i
        if not window_silence[i] and prev:
            e = i-1
            talk_range.append((int(s * step_duration * sample_rate), int(e * step_duration * sample_rate)))
        prev = window_silence[i]
    
    if s >= e:
        talk_range.append((int(s * step_duration * sample_rate), wav_length))

    return len(talk_range), talk_range, sample_rate

def handle_score_1_2(talk_range, short_talk_range=(1, 1.5, 2), sample_rate=0):
    min_talk, med_talk, high_talk = short_talk_range
    min_talk *= sample_rate
    med_talk *= sample_rate
    high_talk *= sample_rate
    lens = talk_range.size 

    if (lens == (talk_range <= min_talk).sum()):
        return "labored", "B", "no trait"
    first_range = (talk_range >= min_talk) & (talk_range < med_talk)
    if first_range.sum() >= 4:
        return "staccato", "B", "A"
    else:
        second_range = (talk_range >= med_talk) & (talk_range < high_talk)
        num_second_range = second_range.sum()
        if num_second_range >= 2:
            return "staccato", "B", "C"
        else:
            high_range = talk_range >= high_talk
            if high_range.sum() >= 1:
                return "staccato", "B", "F"
            else:
                return "discontinuous", "B", "G"
    
def fluency_detecting(short_silence_range=(0.2, 1), silence_threshold=3e-4, step_duration=0.02, very_short_talk_range=(0.4, 1), sample_rate=None, samples=None):
    min_silence, _ = short_silence_range
    n, cut_ranges, sample_rate = extract_silence(min_silence, silence_threshold, step_duration, sample_rate=sample_rate, samples=samples)
    skeleton, trace = derivative(samples, sample_rate, cut_ranges)
    if n == 1:
        talk_range = np.array([(e - s) for s, e in cut_ranges])
        count_hesitation = (talk_range < (very_short_talk_range[0] * sample_rate)).sum()
        score = "No pause at all" if not skeleton else "staccato"
        return score, [], [ [s/sample_rate, e/sample_rate] for s, e in cut_ranges], "no pause" if not skeleton else "skeleton", "", int(count_hesitation), trace
    score = "No idea"
    reason = ""
    trait = ""
    talk_range = np.array([(e - s) for s, e in cut_ranges])
    starts = [s / sample_rate for  s, _ in cut_ranges]
    ends = [e / sample_rate for  _, e in cut_ranges]
    pause_range = list(zip(ends[:-1], starts[1:]))

    starts = [s for s, _ in cut_ranges]
    ends = [e for _, e in cut_ranges]
    starts = np.array(starts[1:])
    ends = np.array(ends[:-1])

    count_hesitation = (talk_range < (very_short_talk_range[0] * sample_rate)).sum()

    count_very_short_talk = ((talk_range < (very_short_talk_range[1] * sample_rate)) & (talk_range > (very_short_talk_range[0] * sample_rate))).sum() 

    short_talk_range = (1, 1.5, 2)
    if count_very_short_talk <=2:
        score = "no staccato" if not skeleton else "staccato"
        reason = "A" if not skeleton else "skeleton"
    elif count_very_short_talk <= 9:
       score, reason, trait = handle_score_1_2(talk_range, short_talk_range, sample_rate)
    else:
        score = "labored"
        reason = "C"
    return score, pause_range, [ [s/sample_rate, e/sample_rate] for s, e in cut_ranges], reason, trait, int(count_hesitation), trace

def long_pause_scorer(n):
    if n > 2:
        return 0
    if n == 2:
        return 1
    if n == 1:
        return 2
    return 3 

def speech_scorer(txt):
    if txt == 'labored':
        return 0
    if txt == 'no staccato':
        return 3
    if txt == 'No pause at all':
        return 3
    if txt == 'discontinuous':
        return 1
    if txt == 'staccato':
        return 2
    return 1

def hesitation_scorer(count_hesitation):
    if count_hesitation <= 2: 
        return 5
    elif count_hesitation <= 5:
        return 4
    return 3

def audio_scorer(fname, text, ratio=((0.85, 1.0), (0.7, 0.85)), dry_run=False):
    if fname.endswith('.mp3'):
        sample_rate, samples = read_mp3(fname)
        # sample_rate, samples = wavfile.read(filename=fname, mmap=True)
    else:
        sample_rate, samples = wavfile.read(filename=fname, mmap=True)
    n, _, _ = extract_silence(min_silence_length=1.0, silence_threshold=1e-4, step_duration=0.005, sample_rate=sample_rate, samples=samples)
    long_pauses = n-1
    long_pauses_score = long_pause_scorer(long_pauses)

    speech_score_text, pause_range, talk_range, reason, trait, count_hesitation, skeleton_trace = fluency_detecting(sample_rate=sample_rate, samples=samples)
    speech_score = speech_scorer(speech_score_text)
    overall_score = min(long_pauses_score, speech_score)

    w_score, w_rate = "NA", "NA"
    ideal_length, student_length = "NA", "NA"

    if overall_score == 3:
        w_score, w_rate, _, _, ideal_length, student_length = syllable.speech_rate_syllable(talk_range, text, ratio=ratio, wpm_avg=0.4)

        hesitation_score = hesitation_scorer(count_hesitation)

        overall_score = min(w_score, hesitation_score)

    if dry_run:
        return overall_score
    return overall_score, long_pauses, speech_score_text, w_score, pause_range, talk_range, reason, trait, w_rate, ideal_length, student_length, count_hesitation, skeleton_trace

text = 'For any marketing course that requires the development of a marketing plan, such as Marketing Management, Marketing Strategy and Segmentation Support Marketing, this is the only planning handbook that guides students through the step-by-step creation of a customized marketing plan while offering commercial software to aid in the process.'

def multiple_overall(input_dir, out_f):
    filenames = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if i.endswith('.wav')]
    r = []
    for f in tqdm(filenames):
        overall_score, long_pauses, speech_score_text, w_score, pause_range, talk_range, reason, trait, w_rate, ideal_length, student_length, count_hesitation, skeleton_trace = audio_scorer(f, text, dry_run=False)
        fname = f.split('/')[-1].split('.')[0]
        j = {
            'filename': fname,
            'score': overall_score,
            'long_pauses': long_pauses,
            'speech_score': speech_score_text,
            'rate score': w_score,
            'pause range': pause_range,
            'talk range': talk_range,
            'reason': reason,
            'trait': trait,
            'ideal length': ideal_length,
            'student length': student_length,
            'ratio': w_rate,
            'no. hesitation': count_hesitation,
            'skeleton trace': skeleton_trace 
        }
        r.append(j)
        with open (os.path.join('./output', f'{fname}.json'), 'w') as output:
            json.dump(j, output)
    
    df = pd.DataFrame(r)
    df.to_csv(out_f)
    
# multiple_overall('./data', out_f='./result_overall_skeletoncheck.csv')
# start = timeit.default_timer()
# print(audio_scorer("./data/f7a6a796-eff7-4a45-b56c-5aa0d9a2376c.mp3", text, False))
# stop = timeit.default_timer()
# print(stop-start)
    
def main(file, text=None, task="RA"):
    if task == 'RA':
        return audio_scorer(fname=file, text=text)
    elif task == 'RS':
        return audio_scorer(fname=file, text=text)
    elif task == 'DI':
        return audio_scorer(fname=file, text=text, ratio=((0.85, 1), (0.7, 0.85)))
    elif task == 'RL':
        return audio_scorer(fname=file, text=text, ratio=((0.85, 1), (0.7, 0.85)))
    
main('abc.wav', text=text, task='RS')
