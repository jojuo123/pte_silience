import spacy
from keyword_spacy import KeywordExtractor
import contextualSpellCheck
import nltk

# Load spacy model
nlp = spacy.load("en_core_web_md")
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
        # spell_check += t._.performed_spellCheck
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
                tokens_for_similarity.append(token.lemma)
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
    _, context_tokens, _, _, _ = lemmatize_and_remove_stopwords(context_text, True)
    content_score = content_swt(student_tokens, context_tokens)
    form_score = swt_form_check(sentences, nlp_tokens)
    grammar_score = swt_grammar_check(sentences)
    vocab_score = content_score

    return content_score + form_score + grammar_score + vocab_score

def writing_scorer(task, student_text, context_text, common_dictionary, detail_dictionary):
    if task == 'SWT':
        swt(student_text, context_text)

print(lemmatize_and_remove_stopwords('I drove a car.', keyword_extract=True)[5])

#kho check vu trong bai viet hoa lung tung (vi co named entity)
