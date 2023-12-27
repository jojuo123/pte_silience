import spacy
from spacy_syllables import SpacySyllables

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("syllables", after="tagger", config={"lang": "en_US"})

def syllable_count(text):
    doc = nlp(text)
    data = [(token.text, token._.syllables, token._.syllables_count) for token in doc]

    cnt = 0
    cnt_dot = 0
    word_cnt = 0
    for d in data:
        cnt += d[2] if d[2] is not None else 0
        cnt_dot += 1 if d[0] == '.' else 0
        word_cnt += 1 if d[2] is not None else 0
    if data[-1][0] == '.':
        cnt_dot -= 1
    return data, word_cnt, cnt, cnt_dot

def speech_rate_syllable(talk_range, text=None, wpm_avg=0.4, dot_pause=0.5, spm_avg=0.2, ideal_length=32, ratio=((0.85, 1.0), (0.7, 0.85))):
    # length = samples.shape[0] / sample_rate
    length = 0
    for (s, e) in talk_range:
        length += (e-s) 
    if text is None:
        w_time = s_time = ideal_length
    else:
        _, word_cnt, cnt, cnt_dot = syllable_count(text)
        w_time = word_cnt * wpm_avg + cnt_dot * dot_pause
        s_time = cnt * spm_avg + cnt_dot * dot_pause
    w_rate = length / w_time

    w_score = 0
    if ratio[0][0] <= w_rate <= ratio[0][1]:
        w_score = 4
    elif ratio[1][0] <= w_rate <= ratio[1][1]:
        w_score = 5
    else:
        w_score = 3
    s_rate = length / s_time
    s_score = 0
    if ratio[0][0] <= s_rate <= ratio[0][1]:
        s_score = 4
    elif ratio[1][0] <= s_rate <= ratio[1][1]:
        s_score = 5
    else:
        s_score = 3
    return w_score, w_rate, s_score, s_rate, w_time, length
