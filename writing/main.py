import spacy
from keyword_spacy import KeywordExtractor
import contextualSpellCheck
import nltk
import pickle as pkl
import pandas as pd
import re

# Load spacy model
nlp = spacy.load("en_core_web_sm")
# contextualSpellCheck.add_to_pipe(nlp)
nlp.add_pipe('keyword_extractor', last=True, config={'top_n': 20, "min_ngram": 3, "max_ngram": 3, "strict": False, "top_n_sent": 4})

def lemmatize_and_remove_stopwords(text, only_rm=False, keyword_extract=False):
    # text = text.lower()
    sentences = nltk.sent_tokenize(text)
    tokens = []
    tokens_for_similarity = []
    nlp_tokens = []
    keywords = []
    spell_check = 0
    for sent in sentences:
        # sent = sent.lower()
        t = nlp(sent)
        nlp_tokens.append(t)
        # print(sent)
        # spell_check += len(t._.suggestions_spellCheck)
        # print(t._.suggestions_spellCheck)
        if keyword_extract:
            keywords += t._.keywords
        for token in t:
            if not only_rm:
                if token.dep_ == 'prt' and token.head.pos_ == 'VERB':
                    tokens.append(token.head.lemma_ + ' ' + token.lemma_)
                else:
                    if not token.is_stop:
                        tokens.append(token.lemma_)
            if not token.is_stop:
                tokens_for_similarity.append(token.lemma_)
    keywords = [k[0] for k in keywords]
    return tokens, tokens_for_similarity, sentences, nlp_tokens, keywords, spell_check

def compute_similarity(tokens1, tokens2):
    tokens1 = set(tokens1)
    tokens2 = set(tokens2)
    set_s = tokens1.intersection(tokens2)
    return len(set_s)

def content_swt(student_tokens, context_tokens):
    sim = compute_similarity(student_tokens, context_tokens)
    sim_score = 0
    if sim < 5:
        sim_score = 0
    elif sim < 15:
        sim_score = 1
    else:
        sim_score = 2
    return sim_score

def swt_form_check(sentences, nlp_tokens):
    fail_cnt = 0
    if len(sentences) != 1:
        return 0
    sent = sentences[0]
    if sent[0].islower() or sent[-1] != '.':
        fail_cnt += 1
    if fail_cnt > 0:
        return 0
    if 5 <= len(nlp_tokens[0]) <= 75:
        return 1
    else:
        return 0

def swt_grammar_check(sentences):
    return 2

def swt(student_text, context_text):
    _, student_tokens, sentences, nlp_tokens, _, spell_check = lemmatize_and_remove_stopwords(student_text, True)
    _, context_tokens, _, _, _, _ = lemmatize_and_remove_stopwords(context_text, True)
    content_score = content_swt(student_tokens, context_tokens)
    form_score = swt_form_check(sentences, nlp_tokens)
    grammar_score = swt_grammar_check(sentences)
    vocab_score = content_score

    return content_score + form_score + grammar_score + vocab_score

def di(student_text, common_dictionary, detail_dictionary):
    _, _, _, nlp_tokens, _, _ = lemmatize_and_remove_stopwords(student_text, True)
    tokens = [t.lemma_ for sent in nlp_tokens for t in sent]
    tokens_2 = [t.text for sent in nlp_tokens for t in sent]
    describing_cnt = compute_similarity(tokens, common_dictionary)
    image_ele_cnt = compute_similarity(tokens_2, detail_dictionary)
    if describing_cnt < 3 or image_ele_cnt < 3:
        return 0
    elif describing_cnt < 5 or image_ele_cnt < 5:
        return 1
    else:
        sum_ = describing_cnt + image_ele_cnt
        if 6 <= sum_ <= 8:
            return 1
        elif 9 <= sum_ <= 12:
            return 2
        elif 13 <= sum_ <= 16:
            return 3
        elif 17 <= sum_ <= 21:
            return 4
        else: 
            return 5 
        
def sst(student_text, common_dictionary, detail_dictionary):
    _, _, _, nlp_tokens, _, spell_check = lemmatize_and_remove_stopwords(student_text, True)
    tokens = [t.lemma_ for sent in nlp_tokens for t in sent]
    tokens_2 = [t.text for sent in nlp_tokens for t in sent if not t.is_punct]
    vocab_score = compute_similarity(tokens, common_dictionary)
    content_score = compute_similarity(tokens_2, detail_dictionary)
    if vocab_score < 3:
        vocab_score = 0
    elif vocab_score <= 4:
        vocab_score = 1
    else:
        vocab_score = 2
    
    if content_score < 5:
        content_score = 0
    elif content_score <= 14:
        content_score = 1
    else:
        content_score = 2
    
    #TODO check grammar + spelling
    grammar_score = 2
    spelling_score = spell_check
    if spelling_score == 0:
        spelling_score = 2
    elif spelling_score <= 1:
        spelling_score = 1
    else:
        spelling_score = 2

    form_score = len(tokens_2)
    if form_score < 40 or form_score > 100:
        form_score = 0
    elif 40 <= form_score <= 49 or 71 <= form_score <= 100:
        form_score = 1
    else:
        form_score = 2
    
    return content_score + vocab_score + form_score + grammar_score + spelling_score

def essay_vocab(tokens, sentence_tokens, common_dictionary):
    base_score = 0.0
    cnt = 0
    for s in sentence_tokens:
        if len(s) > 15:
            cnt += 1
    if cnt <= 2:
        base_score = 0.0
    elif cnt <= 4:
        base_score = 0.5
    else:
        base_score = 1.0

    p_score = 0.0
    for key in common_dictionary.keys():
        if key.endswith('Phrasal'):
            flag = False
            for p in common_dictionary[key]:
                if p in tokens:
                    flag = True
                    tokens = tokens.replace(p, '')
            if flag:
                p_score += 0.25

    linguistic_range_score = p_score + base_score
    if linguistic_range_score - int(linguistic_range_score) > 0.5:
        linguistic_range_score = int(linguistic_range_score) + 1
    else:
        linguistic_range_score = int(linguistic_range_score)
    
    vocab_score = 0.0
    for key in common_dictionary.keys():
        if key.endswith('Words'):
            cnt = 0
            for p in common_dictionary[key]:
                if p in tokens:
                    cnt += 1
            if key.startswith('B1') and cnt > 20:
                vocab_score += 0.5
            if key.startswith('B2') and cnt > 15:
                vocab_score += 0.5
            if key.startswith('C1') and cnt > 10:
                vocab_score += 0.5
            if key.startswith('C2') and cnt > 5:
                vocab_score += 0.5
    
    vocab_score = int(vocab_score)
    return vocab_score + linguistic_range_score
                

def essay(student_text, context_text, common_dictionary):
    _, student_tokens, sentences, nlp_tokens, _, spell_check = lemmatize_and_remove_stopwords(student_text, True)
    _, context_tokens, _, _, _, _ = lemmatize_and_remove_stopwords(context_text, True)

    content_score = compute_similarity(student_tokens, context_tokens)
    if content_score < 6:
        content_score = 0
    elif content_score <= 10:
        content_score = 1
    elif content_score <= 15:
        content_score = 2
    else:
        content_score = 3
    
    tokens_2 = [t.text for sent in nlp_tokens for t in sent if not t.is_punct]
    form_score = len(tokens_2)
    if form_score < 120 or form_score > 380:
        form_score = 0
    elif 120 <= form_score <= 199 or 301 <= form_score <= 380:
        form_score = 1
    else:
        form_score = 2
    if student_text.isupper():
        form_score = 0
    
    grammar_score = 2

    spelling_score = spell_check
    if spelling_score == 0:
        spelling_score = 2
    elif spelling_score <= 1:
        spelling_score = 1
    else:
        spelling_score = 2

    tokens = ' '.join([t.lemma_ for sent in nlp_tokens for t in sent])
    sentence_tokens = [[t.lemma_ for t in sent if not t.is_punct] for sent in nlp_tokens]
    vocab_score = essay_vocab(tokens, sentence_tokens, common_dictionary)

    dev_score = 0
    text = student_text.strip()
    p = re.split(r'\n+', text)
    if len(p) == 4:
        dev_score = 1
    cnt = 0
    for w in common_dictionary['Signalling words']:
        if w in student_text:
            cnt += 1
    if cnt > 7:
        dev_score += 1
    
    return content_score + form_score + grammar_score + spelling_score + vocab_score + dev_score


def load_wordbank():
    with open('wordbank.pkl', 'rb') as f:
        db = pkl.load(f)
    return db


def load_detail_dictionary(detail_dictionary_fname, index):
    df = pd.read_csv(detail_dictionary_fname)
    s = df[df['ID'] == index].iloc[0]['CONTENT']
    s = s.split(',')
    return s


def writing_scorer(task, student_text, context_text=None, common_dictionary=None, detail_dictionary=None):
    if task == 'SWT':
        return swt(student_text, context_text)
    elif task == 'DI':
        di_dict = common_dictionary['DI']
        #detail_dictionary = list
        return di(student_text, di_dict, detail_dictionary)
    elif task == 'essay':
        essay_dict = common_dictionary['essay']
        #context_text = de bai
        return essay(student_text, context_text, essay_dict)
    elif task == 'SST':
        #detail_dictionary = list
        sst_dict = common_dictionary['SST']
        return sst(student_text, sst_dict, detail_dictionary)

# print(lemmatize_and_remove_stopwords('I drove a car.', keyword_extract=True)[5])

#kho check vu trong bai viet hoa lung tung (vi co named entity)
text = '''
Armed police have been brought into NSW schools to reduce crime rates and educate students. The 40 School Liaison Police (SLP) officers have been allocated to public and private high schools across the state.

Organizers say the officers, who began work last week, will build positive relationships between police and students. But parent groups warned of potential dangers of armed police working at schools in communities where police relations were already under strain.

Among their duties, the SLPs will conduct crime prevention workshops, talking to students about issues including shoplifting, offensive behavior, graffiti and drugs, and alcohol. They can also advise school principals. One SLP, Constable Ben Purvis, began to work in the inner Sydney region last week, including at Alexandria Park Community School’s senior campus. Previously stationed as a crime prevention officer at The Rocks’ he now has 27 schools under his jurisdiction in areas including The Rocks, Redfern and Kings Cross.

Constable Purvis said the full-time position would see him working on the broader issues of crime prevention. “I am not a security guard”, he said. “I am not there to patrol the school. We want to improve relationships between police and schoolchildren, to have a positive interaction. We are coming to the school and giving them the knowledge to improve their own safety.” The use of fake ID among older students is among the issues he has already discussed with principals.

Parents’ groups responded to the program positively, but said it may spark a range of community reactions. “It is a good thing and an innovative idea and there could be some positive benefits”, Council of Catholic School Parents executive officer Danielle Cronin said. “Different communities will respond to this kind of presence in different ways.”
'''

text_2 = '''
Armed police have been brought into NSW schools to reduce crime rates and educate students. The 40 School Liaison Police (SLP) officers have been allocated to public and private high schools across the state.

Organizers say the officers, who began work last week, will build positive relationships between police and students. But parent groups warned of potential dangers of armed police working at schools in communities where police relations were already under strain.

Among their duties, the SLPs will conduct crime prevention workshops, talking to students about issues including shoplifting, offensive behavior, graffiti and drugs, and alcohol. They can also advise school principals. One SLP, Constable Ben Purvis, began to work in the inner Sydney region last week, including at Alexandria Park Community School’s senior campus. Previously stationed as a crime prevention officer at The Rocks’ he now has 27 schools under his jurisdiction in areas including The Rocks, Redfern and Kings Cross.

Constable Purvis said the full-time position would see him working on the broader issues of crime prevention. “I am not a security guard”, he said. “I am not there to patrol the school. We want to improve relationships between police and schoolchildren, to have a positive interaction. We are coming to the school and giving them the knowledge to improve their own safety.” The use of fake ID among older students is among the issues he has already discussed with principals.
'''

# print(writing_scorer('SWT', 'Armed police have been brought into NSW schools to reduce crime rates and educate students.', text))
# tmp = lemmatize_and_remove_stopwords(text, keyword_extract=True)[-2]
# print(tmp)

wordbank = load_wordbank()
#SST
detail_dict = load_detail_dictionary('./wordbank/example/sst_detail_dict.csv', 1)
print(writing_scorer("SST", student_text="Australia houses is indicated as expensive. The price has increased significantly, and essentially, the rate has changed the living of the population.", common_dictionary=wordbank, detail_dictionary=detail_dict))

#DI
detail_dict = load_detail_dictionary('./wordbank/DI/Images_Elements.csv', 1)
print(writing_scorer("DI", student_text="western australia and northern people always enter territory education.", common_dictionary=wordbank, detail_dictionary=detail_dict))

#essay
print(writing_scorer("essay", text_2, 'Armed police have been brought into NSW schools to reduce crime rates and educate students.', common_dictionary=wordbank))
