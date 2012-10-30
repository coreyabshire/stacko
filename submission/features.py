import competition_utilities as cu
import csv
import datetime
import features
import numpy as np
import pandas as pd
import re

def camel_to_underscores(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

digit_pattern = re.compile(r'[0-9]')
space_pattern = re.compile(r'\s')
upper_pattern = re.compile(r'[A-Z]')
lower_pattern = re.compile(r'[a-z]')
def letter_densities(text):
    '''
    One of the things I remember from one of the papers that I read, I
    believe in one of the Agichtein papers, was that punctuation
    density could be an indicator of quality in social media. For
    instance, if you see a question with 20 or so question marks, it
    was probably asked by someone who didn't care all that much about
    the quality of their question. To help detect this, this function
    will simply return the ratio of the number of characters in the
    text that are punctuation, to the number of characters that are
    not punctuation. One nuance here to consider is about code
    included in the markdown, and the markdown itself. I don't really
    want to count off for markdown formatting and the like. But I want
    to use this same function for both the title and the body. The
    solution I think is to do the filtering of the code and the URL's
    from the markdown, and any other filtering I need to do on that
    outside of this particular function.
    '''
    n = len(text)
    if n == 0:
        return tuple(0.0 for i in range(6))
    d = float(len(digit_pattern.findall(text)))
    s = float(len(space_pattern.findall(text)))
    u = float(len(upper_pattern.findall(text)))
    l = float(len(lower_pattern.findall(text)))
    p = n - d - s - u - l
    return d/n, s/n, u/n, l/n, p/n, u+l > 0 and u/(u+l) or 0.0

##############################################################
###### FEATURE FUNCTIONS
##############################################################

def body_length(data):
    return data["BodyMarkdown"].apply(len)

def num_tags(data):
    return pd.DataFrame.from_dict({"NumTags": [sum(map(lambda x:
                    pd.isnull(x), row)) for row in (data[["Tag%d" % d
                    for d in range(1,6)]].values)] } ) ["NumTags"]

def title_length(data):
    return data["Title"].apply(len)

def user_age(data):
    return pd.DataFrame.from_dict({"UserAge": (data["PostCreationDate"]
            - data["OwnerCreationDate"]).apply(lambda x: x.total_seconds())})

def code_length(data):
    p = re.compile(r'^(?: {4}|\t).*$', re.M)
    f = lambda s: sum(len(m) for m in p.findall(s))
    v = data["BodyMarkdown"].apply(f)
    return pd.DataFrame.from_dict({"CodeLength": v}) ["CodeLength"]

def num_urls(data):
    p = re.compile(r'https?://[^\s]+')
    f = lambda s: len(p.findall(s))
    v = data["BodyMarkdown"].apply(f)
    return pd.DataFrame.from_dict({"NumUrls": v}) ["NumUrls"]

def title_char_densities(data):
    p = re.compile(r'^(?: {4}|\t).*$', re.M)
    tups = data.BodyMarkdown.apply(lambda s: letter_densities(p.sub('', s)))
    pre = 'Body'
    suf = 'Density'
    names = 'Digit,Space,Upper,Lower,Punctuation,UpperLower'.split(',')
    d = {}
    for i in range(len(names)):
        k = '%s%s%s' % (pre,names[i],suf)
        d[k] = tups.apply(lambda x: x[i])
    return pd.DataFrame.from_dict(d)

###########################################################

def extract_features(feature_names, data):
    fea = pd.DataFrame(index=data.index)
    for name in feature_names:
        if name in data:
            fea = fea.join(data[name])
        else:
            fea = fea.join(getattr(features, 
                camel_to_underscores(name))(data))
    return fea

