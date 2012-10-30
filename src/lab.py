import competition_utilities as cu
import numpy as np
import scipy
from features import Body
import features
import nltk
import re
from words import get_words
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from collections import Counter
from onlineldavb import OnlineLDA, parse_doc_list
import time
import pandas as pd
import operator
from sklearn import svm

class Problem():
    def __init__(self):
        self.dataset
        self.features
        self.labels

    def train():
        pass

    def predict():
        pass
    
def istag(x):
    return (x is not None) and (not pd.isnull(x)) and (x != 'nan')

class QuestionSetCSV:

    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'rb')
        self.reader = csv.reader(self.file)
        self.header = self.reader.next()

    def parse_doc(self, row):
        title = row['Title']
        body = row['BodyMarkdown']
        tags = ' '.join(filter(istag, [row['Tag%d' % t] for t in range(1,6)]))
        postid = row['PostId']
        doc = ' '.join([title, body, tags])
        name = postid
        return doc, name
        
    def get_batch(self, size):
        docset = []
        articlenames = []
        for i in range(size):
            try:
                line = self.reader.next()
                row = dict(zip(self.header, line))
                doc, name = self.parse_doc(row)
                docset.append(doc)
                articlenames.append(name)
            except e:
                print e
                pass
        return docset, articlenames

    def close(self):
        self.file.close()

class QuestionSet:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def parse_doc(self, row):
        title = row['Title']
        body = row['BodyMarkdown']
        tags = ' '.join(filter(istag, [row['Tag%d' % t] for t in range(1,6)]))
        postid = row['PostId']
        doc = ' '.join([title, body, tags])
        name = postid
        return doc, name
        
    def parse_doc_no_code(self, row):
        title = row['Title']
        body = features.code_pattern.sub('', row['BodyMarkdown'])
        tags = ' '.join(filter(istag, [row['Tag%d' % t] for t in range(1,6)]))
        postid = row['PostId']
        doc = ' '.join([title, body, tags])
        name = postid
        return doc, name

    def get_batch(self, start, end):
        docset = []
        articlenames = []
        for i in range(start, end):
            try:
                row = self.dataframe.ix[i]
                doc, name = self.parse_doc_no_code(row)
                docset.append(doc)
                articlenames.append(name)
            except e:
                print e
                pass
        return docset, articlenames

    def close(self):
        self.file.close()

    
def compute_y_true(data):
    s = data.OpenStatus
    k = sorted(s.unique())
    m = dict((v,k) for k,v in enumerate(k))
    return np.array(s.apply(m.get))

def measure_model(datasize=1000, testsize=500):
    data = cu.get_sample_data_frame(datasize)
    test = cu.get_test_data_frame(testsize)
    #data = full.ix[len(full)/4:].reset_index() # last n/4 * 3 records
    #test = full.ix[:(len(full)/4)-1].reset_index() # first n/4 records
    #data = cu.get_dataframe('train-sample.csv')
    #test = cu.get_dataframe('public_leaderboard.csv')
    fea = features.extract_features(data)
    test_features = features.extract_features(test)
    rf = RandomForestClassifier(n_estimators=50, verbose=2,
                                compute_importances=True, n_jobs=5)
    rf.fit(fea, data["OpenStatus"])
    probs = rf.predict_proba(test_features)
    new_priors = cu.load_priors('train.csv')
    old_priors = cu.compute_priors(data.OpenStatus)
    probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
    y_true = compute_y_true(test)
    score = multiclass_log_loss(y_true, probs)
    return score, rf, fea

def print_topics(vocab, testlambda, n=10):
    """
    Displays topics fit by Online LDA VB. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    for k in range(0, len(testlambda)):
        lambdak = list(testlambda[k, :])
        lambdak = lambdak / sum(lambdak)
        temp = zip(lambdak, range(0, len(lambdak)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        print 'topic %03d:' % (k)
        # feel free to change the "53" here to whatever fits your screen nicely.
        for i in range(0, n):
            print '%20s  \t---\t  %.4f' % (vocab[temp[i][1]], temp[i][0])
        print

def make_topic_summary(vocab, testlambda, n=10):
    """
    Displays topics fit by Online LDA VB. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    data = {}
    for k in range(0, len(testlambda)):
        lambdak = list(testlambda[k, :])
        lambdak = lambdak / sum(lambdak)
        t = zip(lambdak, range(0, len(lambdak)))
        t = sorted(t, key = lambda x: x[0], reverse=True)
        v = ['%s %.4f' % (vocab[t[i][1]], t[i][0]) for i in range(0, n)]
        data['Topic%03d' % k] = v
    df = pd.DataFrame.from_dict(data)
    return df

def write_topics_csv(vocab, testlambda, csvfile, n=10):
    """
    Displays topics fit by Online LDA VB. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    df = make_topic_summary(vocab, testlambda, n)
    df.to_csv(csvfile)
    return df

def make_topic_columns(lda, data, K, D, batchsize):
    questions = QuestionSet(data)
    allgamma = numpy.zeros((len(data), K))
    for iteration in range(0, len(data) / batchsize):
        start = iteration * batchsize
        end = start + batchsize
        # Download some articles
        (docset, articlenames) = questions.get_batch(start, end)
        # Give them to online LDA
        (gamma, bound) = lda.update_lambda(docset)
        allgamma[start:end,:] = gamma
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = parse_doc_list(docset, lda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, lda._rhot, numpy.exp(-perwordbound))
    # copy to dataframe
    for k in range(K):
        data['Topic%d'%k] = allgamma[:,k]

def measure_lda(datasize=1000, testsize=500):
    # The number of documents to analyze each iteration
    batchsize = 100
    # The total number of questions on Stack Overflow
    D = 3.3e6
    # The number of topics
    K = 100
    # How many documents to look at
    documentstoanalyze = datasize / batchsize
    # Our vocabulary
    vocab = [w.strip() for w in file('./vocab2.txt')]
    W = len(vocab)
    # the data
    data = cu.get_sample_data_frame(datasize)
    test = cu.get_test_data_frame(testsize)
    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    lda = OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
    make_topic_columns(lda, data, K, D, batchsize)
    make_topic_columns(lda, test, K, D, batchsize)
    
    #data = full.ix[len(full)/4:].reset_index() # last n/4 * 3 records
    #test = full.ix[:(len(full)/4)-1].reset_index() # first n/4 records
    #data = cu.get_dataframe('train-sample.csv')
    #test = cu.get_dataframe('public_leaderboard.csv')

    fea = features.extract_features(data)
    test_features = features.extract_features(test)
    rf = RandomForestClassifier(n_estimators=50, verbose=2, compute_importances=True, n_jobs=5)
    rf.fit(fea, data["OpenStatus"])
    probs = rf.predict_proba(test_features)
    new_priors = cu.load_priors('train.csv')
    old_priors = cu.compute_priors(data.OpenStatus)
    probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
    y_true = compute_y_true(test)
    score = multiclass_log_loss(y_true, probs)
    return score, rf, fea, data, datagamma

def learning_curve():
    n = 50000
    nsteps = 10
    full = cu.get_sample_data_frame(n)
    data = full.ix[0:int(n*.6)-1].reset_index()
    cval = full.ix[int(n*.6):int(n*.8)-1].reset_index()
    test = full.ix[int(n*.8):n-1].reset_index()
    step = len(data) / nsteps
    ndata = len(data)
    mvec = range(step, ndata + step, step)
    test_features = features.extract_features(test)
    data_error = []
    cval_error = []
    for i in range(len(mvec)):
        m = mvec[i]
        print 'running for size', m
        train = data.ix[0:m-1].reset_index()
        fea = features.extract_features(train)
        rf = RandomForestClassifier(n_estimators=50, verbose=0, compute_importances=False, n_jobs=5)
        rf.fit(fea, train["OpenStatus"])
        new_priors = cu.load_priors('train.csv')
        old_priors = cu.compute_priors(train.OpenStatus)
        # predict train
        probs = rf.predict_proba(fea)
        #probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
        y_true = compute_y_true(train)
        score = multiclass_log_loss(y_true, probs)
        data_error.append(score)
        # predict cval
        probs = rf.predict_proba(test_features)
        #probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
        y_true = compute_y_true(test)
        score = multiclass_log_loss(y_true, probs)
        cval_error.append(score)
    return mvec, data_error, cval_error

def n_estimators_curve():
    n = 10000
    nsteps = 10
    maxest = 200
    full = cu.get_sample_data_frame(n)
    data = full.ix[0:int(n*.6)-1].reset_index()
    cval = full.ix[int(n*.6):int(n*.8)-1].reset_index()
    test = full.ix[int(n*.8):n-1].reset_index()
    step = maxest / nsteps
    ndata = len(data)
    nvec = range(step, maxest + step, step)
    test_features = features.extract_features(test)
    data_error = []
    cval_error = []
    train = data.reset_index()
    fea = features.extract_features(train)
    for i in range(len(nvec)):
        n_est = nvec[i]
        print 'running for n_estimators', n_est
        rf = RandomForestClassifier(n_estimators=n_est, verbose=0, compute_importances=False, n_jobs=5)
        rf.fit(fea, train["OpenStatus"])
        #new_priors = cu.load_priors('train.csv')
        #old_priors = cu.compute_priors(train.OpenStatus)
        # predict train
        probs = rf.predict_proba(fea)
        #probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
        y_true = compute_y_true(train)
        score = multiclass_log_loss(y_true, probs)
        data_error.append(score)
        # predict cval
        probs = rf.predict_proba(test_features)
        #probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
        y_true = compute_y_true(test)
        score = multiclass_log_loss(y_true, probs)
        cval_error.append(score)
    return nvec, data_error, cval_error

def measure_bayes(datasize=1000, testsize=500):
    data = cu.get_sample_data_frame(datasize)
    test = cu.get_test_data_frame(testsize)
    nbfd = features.naive_features(data)
    nbft = features.naive_features(test)
    nb = nltk.NaiveBayesClassifier.train(nbfd)
    probs = []
    for i in range(len(nbft)):
        p = nb.prob_classify(nbft[i][0])
        probs.append([p.prob(s) for s in p.samples()])
    probs = np.array(probs)
    new_priors = cu.load_priors('train.csv')
    old_priors = cu.compute_priors(data.OpenStatus)
    probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
    y_true = compute_y_true(test)
    score = multiclass_log_loss(y_true, probs)
    return score, nb, y_true, probs

def measure_prior(datasize=1000, testsize=500):
    data = cu.get_sample_data_frame(datasize)
    test = cu.get_test_data_frame(testsize)
    priors = cu.load_priors('train.csv')
    num_samples = len(test)
    probs = np.kron(np.ones((num_samples,1)), priors)
    y_true = compute_y_true(test)
    score = multiclass_log_loss(y_true, probs)
    return score

def raw_accuracy(y_true, y_pred):
    pass

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float

    Thanks to someone on the forum.
    """
    
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota


def write_topics_for_status(status):
    '''
    One of my analysis questions is how different do the topics
    look for each of the different status that I have. To help
    with this, I wrote this little macro like procedure to make
    it easier to pull all the topics for a particular status.
    '''
    data = cu.get_sample_data_frame_by_status(status)
    lda = OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
    make_topic_columns(lda, data, K, D, batchsize)
    write_topics_csv(vocab, lda._lambda, 'topics.csv', 10)

def topic_summary_by_status(vocab, K, D, n, limit, batchsize):
    '''
    The purpose of this procedure is to write out a CSV that compares
    all the different topics for all the different categories in the
    main sample.
    '''
    n = 10
    pieces = []
    for status in cu.labels:
        data = cu.get_sample_data_frame_by_status(status, limit)
        lda = OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
        make_topic_columns(lda, data, K, D, batchsize)
        summary = make_topic_summary(vocab, lda._lambda, n)
        pieces.append(summary)
    return pd.concat(pieces, keys=cu.labels)

def extract_text(data):
    '''
    '''
    pass

def extract_sentences(self):
    sentences = []
    for block in self.get_blocks('text'):
        block = url_pattern.sub('', block)
        for sentence in punkt_tokenizer.tokenize(block):
            sentences.append(sentence.strip())
    return sentences

def extract_words_from_row(row):
    '''
    This function turns a given row of data for this particular type
    of problem into a simple vector of words.
    '''
    title = row['Title']
    body = row['BodyMarkdown']
    tags = ' '.join(filter(istag, [row['Tag%d' % t] for t in range(1,6)]))
    postid = row['PostId']
    text = ' '.join([title, body, tags])
    return get_words(text)

def sort_vocabulary_dict(d):
    return dict((k, sorted(v.items(),
                           key=operator.itemgetter(1),
                           reverse=True))
                for k, v in d.items())

def word_ratio(dc, uw):
    r = defaultdict(lambda: defaultdict(float))
    for c in dc.keys():
        for w in uw[c].keys():
            r[c][w] = float(uw[c][w]) / float(dc[c])
    return r

def mychi2(a, b, e):
    return (math.pow(a - e, 2) / e) + (math.pow(b - e, 2) / e)

def word_chisqtest(w, uw):
    a = float(uw['open'][w])
    b = float(uw['closed'][w])
    e = float(uw['total'][w]) / 2.0
    c2 = (math.pow(a - e, 2) / e) + (math.pow(b - e, 2) / e)
    return c2, a, b, e

def extract_vocabulary(data):
    '''
    This function processes the data given to extract the vocabulary
    to be used by the various analysis routines I'm using for the
    contest. It applies the heuristics I was using to manually prepare
    the vocabulary file. I decided it would be better if all of those
    notions were instead captured in a reusable function so that I
    could reproduce my work later, and so that it would be easier to
    adjust the size of the vocabulary and run various tests with those
    adjustments.
    '''
    vocab = []
    # word count and unique word count by category and totals
    dc = defaultdict(int)
    nc = defaultdict(int)
    uc = defaultdict(int)
    nw = defaultdict(lambda: defaultdict(int))
    uw = defaultdict(lambda: defaultdict(int))
    for d in range(len(data)):
        ws = extract_words_from_row(data.ix[d])
        c = data.OpenStatus[d]
        dc[c] += 1
        for xc, xw, wx in [(nc, nw, ws), (uc, uw, set(ws))]:
            for w in wx:
                xc['total'] += 1
                xc[c] += 1
                xw['total'][w] += 1
                xw[c][w] += 1
                if c != 'open':
                    xc['closed'] += 1
                    xw['closed'][w] += 1
    vocab = nw['total'].keys()
    nwx = sort_vocabulary_dict(nw)
    uwx = sort_vocabulary_dict(uw)
    return dc, nc, uc, nw, uw, nwx, uwx

def filter_vocabulary(uw):
    words = [k for k,v in uwx['total']]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    vocab = [w for w in words if w not in stopwords and len(w) > 2]
    return vocab

def run_job(f, note=None):
    '''
    I often have to run long jobs when working on machine learning
    algorithms. One of the things I can do to help increase my
    productivity is work on things in between long running
    functions. To help with this, I will use this function to run
    long-running jobs. It will record how long the job takes to run,
    print some useful time and date information, print how long it
    took to run it, return the same results, and beep when its
    complete, and give me a note optionally to remind me what I was
    working on so I know what to do next.
    '''
    start = time.clock()
    print 'Started at', time.asctime()
    result = f()
    finish = time.clock()
    print 'Finished at', time.asctime()
    print 'Took %.2f seconds.' % (finish - start)
    if note:
        print 'Note:', note
    print '\a'
    return result

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

def get_svm_word_indices(vidx, text):
    x = np.zeros(len(vidx), dtype=numpy.double)
    for w in get_words(text):
        if w in vidx:
            x[vidx[w]] = 1.0
    return scipy.sparse.csr_matrix(x)

def make_svm_extract(vidx):
    return lambda text: get_svm_word_indices(vidx, text)
        
def extract_svm_features(vidx, data):
    x = scipy.sparse.lil_matrix((len(data),len(vidx)))
    for i in range(len(data.BodyMarkdown)):
        for w in get_words(data.BodyMarkdown[i]):
            if w in vidx:
                x[i,vidx[w]] = 1.0
    return scipy.sparse.csr_matrix(x)

def get_vocab_index_lookup(vocab):
    '''
    Build a dictionary of words to index for the given word list.
    '''
    return dict((w,i) for i,w in enumerate(vocab))

def measure_svm(datasize=1000, testsize=500):
    data = cu.get_sample_data_frame(datasize)
    test = cu.get_test_data_frame(testsize)
    vocab = [w.strip() for w in file('vocab4.txt')][0:1000]
    vidx = get_vocab_index_lookup(vocab)
    print 'extracting data features'
    xdata = extract_svm_features(vidx, data)
    print 'extracting test features'
    xtest = extract_svm_features(vidx, test)
    labels = sorted(cu.labels)
    ydata = data.OpenStatus.apply(labels.index).tolist()
    model = svm.sparse.SVC(probability=True)
    print 'fitting model'
    model.fit(xdata, ydata)
    print 'rest'
    probs = model.predict_proba(xtest)
    new_priors = cu.load_priors('train.csv')
    old_priors = cu.compute_priors(data.OpenStatus)
    probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
    y_true = compute_y_true(test)
    score = multiclass_log_loss(y_true, probs)
    return score, model
