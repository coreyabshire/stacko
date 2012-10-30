# Word tokenization routines.

import nltk
import re

punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
code_pattern = re.compile(r'^(?: {4}|\t).*$', re.M)
url_pattern = re.compile(r'https?://[^\s]+')

code_word_1 = re.compile(r'(.)([A-Z][a-z]+)')
code_word_2 = re.compile(r'([a-z0-9])([A-Z])')
code_word_3 = re.compile(r'[-_/:?]')
split_pattern = re.compile(r'[^a-zA-Z#+]')

def split_code_word(name):
    '''
    This is a simple routine to split a given word into multiple words
    based on whether it contains camelcase or underscores, as is
    typically done in programming.
    '''
    s = code_word_1.sub(r'\1 \2', name)
    s = code_word_2.sub(r'\1 \2', s)
    s = code_word_3.sub(r' ', s)
    return split_pattern.split(s)

def get_sentences(text):
    return [s.strip() for s in punkt_tokenizer.tokenize(text)]
    
def get_words_from_sentence(sentence):
    words = []
    for word in nltk.word_tokenize(sentence):
        for subword in split_code_word(word):
            subword = subword.strip()
            if subword != '':
                words.append(subword)
    return words

special = set(['c', 'f'])
def rejoin_special(ws):
    words = []
    n = len(ws)
    i = 0
    while i < n:
        a = ws[i]
        b = (i + 1) < n and ws[i + 1] or ''
        if b == '#' and a in special:
            words.append('%s%s' % (a,b))
            i += 2
        else:
            words.append(a)
            i += 1
    return words

def get_words(text):
    '''
    This is the primary interface function. It expects to receive a
    block of markdown from the dataset and will return all the words
    we care about from that block of text. It handles the interface to
    NLTK, using the sentence splitting routines and anything else that
    may be required to split the text up nicely and filter any stuff
    that we don't care to consider as words that may be specific to
    this particular problem domain.
    '''
    words = []
    text = code_pattern.sub(' ', text)
    text = url_pattern.sub(' ', text)
    for sentence in get_sentences(text):
        ws = get_words_from_sentence(sentence)
        ws = rejoin_special(ws)
        for word in ws:
            word = word.lower()
            words.append(word)
    return words
