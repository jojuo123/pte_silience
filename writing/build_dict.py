import pandas as pd
import os

import spacy
from keyword_spacy import KeywordExtractor
import contextualSpellCheck
import nltk
import pickle as pkl

# Load spacy model
nlp = spacy.load("en_core_web_md")
# contextualSpellCheck.add_to_pipe(nlp)
nlp.add_pipe('keyword_extractor', last=True, config={'top_n': 20, "min_ngram": 3, "max_ngram": 3, "strict": False, "top_n_sent": 4})

def lemmatize(text):
    t = nlp(text)
    t = [t_.lemma_ for t_ in t]
    return ' '.join(t)

tasks = ['DI', 'essay', 'SST']
dictionary = {}
for t in tasks:
    l_file = os.listdir('wordbank/' + t)
    l_file = [l for l in l_file if l.endswith('.csv')]
    for f in l_file:
        if t == 'DI':
            if f.endswith('Images_Elements.csv'):
                continue
        df = pd.read_csv('wordbank/' + t + '/' + f)
        col = df.columns[0]
        content = df[col].tolist()
        if t == 'SST':
            content = [lemmatize(c) for c in content]
        if not dictionary.__contains__(t):
            dictionary[t] = {}
        dictionary[t][col] = content

with open('wordbank.pkl', 'wb') as f:
    pkl.dump(dictionary, f)