import os
import competition_utilities as cu
import csv
import datetime
import features
import numpy as np
import pandas as pd
import re
import pickle
import nltk
from collections import Counter
from collections import defaultdict

#with open('url.regex') as f:
#    url_pattern = re.compile(f.read())

punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
code_pattern = re.compile(r'^(?: {4}|\t).*$', re.M)

def matches_len(s, p):
    '''Count the lines of code in the given markdown text.'''
    return sum(len(m) for m in p.findall(s))

def matches_count(s, p):
    return len(p.findall(s))

def code_len(s):
    return matches_len(s, code_pattern)

def num_urls(s):
    return matches_count(s, url_pattern)

def is_code_line(line):
    '''Return true if the line of markdown text is a code line.'''
    return line.startswith('    ') or line.startswith('\t')

def classify_line_for_block(line):
    '''Return 'code' or 'text', depending on the type of line given.'''
    if is_code_line(line):
        return 'code'
    else:
        return 'text'

def get_urls(text):
    return [m[0] for m in url_pattern.findall(text)]

def split_blocks(text):
    lines = text.splitlines()
    block_type = ''
    blocks = []
    block = []
    def collect_block(block_type, block):
        if block:
            block_text = '\n'.join(block)
            if block_text != '':
                blocks.append((block_type, block_text))
    for line in text.splitlines():
        line_type = classify_line_for_block(line)
        if line_type == block_type:
            block.append(line)
        else:
            collect_block(block_type, block)
            block_type = line_type
            block = [line]
    collect_block(block_type, block)
    return blocks

class Body():

    def __init__(self, text):
        self.text = text
        self.blocks = split_blocks(text)

    def get_blocks(self, block_type):
        return [v for t,v in self.blocks if t == block_type]

    def get_text_urls(self):
        return get_urls(self.get_all_text())

    def has_urls(self):
        return len(self.get_text_urls()) > 0

    def count_code_blocks(self):
        return sum([1 for t, v in self.blocks if t == 'code'])
        
    def get_all(self, block_type):
        return '\n'.join(v for t,v in self.blocks if t == block_type)
        
    def get_all_text(self):
        return self.get_all('text')
                
    def get_all_code(self):
        return self.get_all('code')

    def count_lines_of_code(self):
        return len(self.get_all_code().splitlines())
    
    def get_sentences(self):
        sentences = []
        for block in self.get_blocks('text'):
            block = url_pattern.sub('', block)
            for sentence in punkt_tokenizer.tokenize(block):
                sentences.append(sentence.strip())
        return sentences
    
    def get_words(self):
        split_pattern = re.compile(r'[^a-zA-Z]')
        words = []
        for sentence in self.get_sentences():
            for word in nltk.word_tokenize(sentence):
                for word2 in split_pattern.split(word):
                    word2 = word2.strip()
                    if word2 != '':
                        words.append(word2)
        return words
        
    def get_word_freq(self):
        return nltk.FreqDist([w.lower() for w in self.get_words()])

    def get_unique_words(self):
        return [w for w in self.get_word_freq()]

    def __repr__(self):
        return self.text

def interest_ratio(wc):
    a = float(wc['open'])
    b = float(sum(v for k,v in wc.iteritems() if k != 'open'))
    if a + b == 0:
        return 0.5
    else:
        return b / (a + b)

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-float(z)))

def interest_score(wc):
    n = sum(v for k,v in wc.iteritems() if k != 'open')
    return interest_ratio(wc) * sigmoid(float(n) * 0.1 - 10.0)

def interesting_word_hash(nb):
    return dict((k,interest_score(v)) for k,v in nb.words.iteritems())

def highest_interest_words(interest):
    return sorted(interest, key=interest.get)

def print_highest_interest(nb, n=1000):
    interest = interesting_word_hash(nb)
    highest = highest_interest_words(interest)
    words = highest[-n:]
    for w in words.reverse():
        wq = '"%s"' % w
        print '\t'.join(str(x) for x in [
                wq,
                interest[w],
                nb.words[w]['open'],
                sum(v for k,v in nb.words[w].iteritems() if k != 'open'),
                nb.words[w]['not a real question'],
                nb.words[w]['off topic'],
                nb.words[w]['not constructive'],
                nb.words[w]['too localized']])              

def write_highest_words(highest, n=1000):
    with open('highest.txt', 'w') as f:
        for w in highest[-n:]:
            f.write('%s\n' % w)

def sigmoid_table(a, b, inc):
    i = a
    while i <= b:
        print i, sigmoid(i)
        i = i + inc

def read_texts_by_label(data, labels=cu.labels):
    '''Make the text in the bodies more convenient to work with.'''
    texts_by_label = {}
    for label in labels:
        texts = []
        subset = data[data.OpenStatus == label]
        for text in subset.BodyMarkdown:
            texts.append(text)
        texts_by_label[label] = texts
    return texts_by_label

def camel_to_underscores(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

##############################################################
###### FEATURE FUNCTIONS
##############################################################

def body_length(data):
    return data["BodyMarkdown"].apply(len)

def num_tags(data):
    return pd.DataFrame.from_dict({"NumTags": [sum(map(lambda x:
                    pd.isnull(x), row)) for row in (data[["Tag%d" % d
#                    x != 'nan', row)) for row in (data[["Tag%d" % d
                    for d in range(1,6)]].values)] } ) ["NumTags"]

def title_length(data):
    return data["Title"].apply(len)

def user_age(data):
    return pd.DataFrame.from_dict({"UserAge": (data["PostCreationDate"]
            - data["OwnerCreationDate"]).apply(lambda x: x.total_seconds())})

def log_reputation_at_post_creation(data):
    return np.log(data["ReputationAtPostCreation"])

def code_length(data):
    p = re.compile(r'^(?: {4}|\t).*$', re.M)
    f = lambda s: sum(len(m) for m in p.findall(s))
    return data["BodyMarkdown"].apply(f)

def num_urls(data):
    p = re.compile(r'https?://[^\s]+')
    f = lambda s: len(p.findall(s))
    return data["BodyMarkdown"].apply(f)
        


###########################################################

def get_extra_features(data):
    extra = {
        #"HasUrls": [],
        #"NumUrls": [],
        #"NumCodeBlocks": [],
        "NumCodeLines": [],
        #"NumBodyWords": []
        }
    n = len(data)
    count = 0
    for text in data.BodyMarkdown:
        #print '%5d of %5d (%3f pct) %4d' % (count, n, float(count)/float(n), len(text))
        body = Body(text)
        #extra["HasUrls"].append(body.has_urls())
        #extra["NumUrls"].append(len(body.get_text_urls()))
        #extra["NumCodeBlocks"].append(body.count_code_blocks())
        extra["NumCodeLines"].append(body.count_lines_of_code())
        #extra["NumBodyWords"].append(len(body.get_unique_words()))
        count += 1
    return pd.DataFrame.from_dict(extra)

def all_tags(data):
    tags = defaultdict(int)
    fields = ['Tag%d' % i for i in range(1,6)]
    for i in range(len(data)):
        for f in fields:
            t = data[f][i]
            if not pd.isnull(t):
                tags[t] += 1
    return tags

def naive_features(data):
    features = []
    fields = ['Tag%d' % i for i in range(1,6)]
    for i in range(len(data)):
        tags = {}
        label = data['OpenStatus'][i]
        for f in fields:
            tag = data[f][i]
            if not pd.isnull(tag):
                tags[tag] = 1
        features.append((tags, label))
    return features

def extract_features(data):
    feature_names = [ "BodyLength"
                    , "NumTags"
                    , "OwnerUndeletedAnswerCountAtPostTime"
                    , "ReputationAtPostCreation"
                    , "TitleLength"
                    , "UserAge"
                    ]
    n = len(data)
    fea = pd.DataFrame(index=data.index)
    feature_names.extend(filter(lambda c: c.startswith('Topic'), data.columns))

    #print 'opening highest.txt'
    #with open('highest.txt', 'r') as f:
    #    highest = set(f.read().splitlines())
    #print 'read all %s highest words' % len(highest)
    #if os.path.exists('wordsets.pickle'):
    #    print 'loading wordsets'
    #    with open('wordsets.pickle', 'r') as f:
    #        wordsets = pickle.load(f)
    #print 'computing word sets'
    #wordsets = [set(Body(s).get_unique_words()) for s in data.BodyMarkdown]
    #print 'dumping wordsets pickle'
    #with open('wordsets.pickle', 'w') as f:
    #    pickle.dump(wordsets, f)
    #print 'computed %s word sets' % len(wordsets)
    #print 'extracting highest word based features'
    #for word in highest:
    #    fea[word] = pd.DataFrame.from_dict({word: [word in wordsets[i] for i in range(n)]})[word]
    #print 'extracting other features'
    for name in feature_names:
        if name in data:
            fea = fea.join(data[name])
        else:
            fea = fea.join(getattr(features, 
                camel_to_underscores(name))(data))
    #re_qm = re.compile('\?')
    #fea['HasBodyQM'] = data.BodyMarkdown.apply(lambda b: re_qm.search(b) != None)
    #fea['IsNoob'] = data.ReputationAtPostCreation <= 1
    #fea['IsLeech'] = data.OwnerUndeletedAnswerCountAtPostTime == 0
    #print 'Added HasBodyQM: ', Counter(fea['HasBodyQM'])
    #print 'Added IsNoob: ', Counter(fea['IsNoob'])
    #print 'Added IsLeech: ', Counter(fea['IsLeech'])
    #print "generating extra features"
    #fea = fea.join(get_extra_features(data))
    
    return fea

if __name__=="__main__":
              
    data = cu.get_dataframe("C:\\Projects\\ML\\stack\\data\\train-sample.csv")
    features = extract_features(data)
    print(features)


def get_words(data, i):
    p = re.compile('^[a-z][a-z-]*[a-z]$')
    body = Body(data.BodyMarkdown[i])
    words = body.get_unique_words()
    punks = [w for w in words if not p.match(w)]
    stops = [w for w in words if w in stopwords]
    words = [w for w in words if not w in stopwords and p.match(w)]
    return words, punks, stops

