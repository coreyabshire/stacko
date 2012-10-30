# word.py: Word tokenization routines for the Predicting Closed
# Questions on Stack Overflow competition on Kaggle.

# One of the key pieces of the machine learning pipeline for this
# particular problem domain is that of breaking up the body of the
# questions into features that we can use to train the
# algorithms. This module implements that piece of functionality. It
# makes heavy use of both the NLTK and RE modules to do the heavy
# lifting, adding a few nuances required to be effective in this
# domain.

import nltk
import re

# The words in the body of the questions are fairly typical English
# text, so the main tokenization is performed via NLTK.

# Using the punkt_tokenizer lets us split the body text blocks in
# the body into sentences somewhat reliably. This is needed to feed
# the word tokenizer later on.
punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Since the body of the questions is in markdown, and since these 
# are questions for a programming Q&A site, there will likely be code
# in many of the questions. Whether or not to include code as tokens
# for LDA or other feature sets was one of the things I struggled with
# during the competition. The following regular expression is what I 
# used to find and strip out if desired the code from the questions.
# In the final model, I used the length of all code in characters
# as a feature, but did not use the words in the code.
# In markdown, code is simply prefixed by 4 spaces or a single tab.
code_pattern = re.compile(r'^(?: {4}|\t).*$', re.M)

# Another unique aspect of the programming Q&A nature of these questions
# is that they are likely to include URL's referencing other sites. I found
# that the tokenizer really didn't like URL's all that much and didn't see
# much use in having the individual words of a URL as features anyway, so
# I ended up stripping them out before tokenization. I did later use them
# as a feature, by simply providing a count of the URL's found as a feature.
url_pattern = re.compile(r'https?://[^\s]+')

# The following 4 regular expressions support the split_code_word function.
# TODO: see if there's a way to move them into the body of the function.
# The function and these regular expressions are based on an observation that
# whenever words that resemble programming tokens such as variable names
# or function names are found in the non-code portions of the question body
# they are often times composed of multiple real words that may add value
# as individual features. This function and the regular expressions split
# programming tokens (a.k.a. "code words") into their component words. They 
# follow the three most common patterns I could think of: camel case and 
# separation by underscores, hypens, or a few other punctuation.
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
    '''
    The first level of parsing for the body text. In order for the word
    tokenization to work properly within NLTK, the text must first be split
    into sentences. NLTK provides an advanced algorithm for doing sentence
    splitting that can be / has been trained on other natural language text.
    One of the things I learned in the competition about NLP is that
    splitting words and sentences is more complex than I had originally 
    thought. For instance, it can include things like abbreviations as cases
    where it may not be clear whether its really a sentence or not. Rather
    than build my own machine learning approach to this problem, I just used
    NLTK throughout.
    '''
    return [s.strip() for s in punkt_tokenizer.tokenize(text)]
    
def get_words_from_sentence(sentence):
    '''
    Given a sentence, this function will break it up into words. It goes
    further than the normal NLTK word tokenizer, by also splitting up certain
    words we call "code words" into their component words. This means words
    like HereAreSomeWords will be split up into Here Are Some Words. This is
    useful in this context, because such words found outside their code blocks
    in the markdown may prove to be some of the more useful words to train on.
    '''
    words = []
    for word in nltk.word_tokenize(sentence):
        for subword in split_code_word(word):
            subword = subword.strip()
            if subword != '':
                words.append(subword)
    return words

special = set(['c', 'f'])
def rejoin_special(ws):
    '''
    This function handles a special case I thought could be significant
    for this particular problem domain. Since the most predominant language
    on Stack Overflow is C#, and because the tokenizer will by default
    split C# into two independent tokens, I felt it would be worthwhile
    to join this particular token pair back into a single token. I also 
    did F# since it is another language found on there and was relatively
    easy to add as well.
    '''
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
