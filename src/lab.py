import competition_utilities as cu
import numpy as np
from features import Body
import features
import nltk
import re
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from collections import Counter
import onlineldavb

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
    return (x is not None) and (x != 'nan')

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
        
    def get_batch(self, start, end):
        docset = []
        articlenames = []
        for i in range(start, end):
            try:
                row = self.dataframe.ix[i]
                doc, name = self.parse_doc(row)
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

def make_topic_columns(olda, data, K, batchsize):
    questions = QuestionSet(data)
    allgamma = numpy.zeros((len(data), K))
    for iteration in range(0, len(data) / batchsize):
        start = iteration * batchsize
        end = start + batchsize
        # Download some articles
        (docset, articlenames) = questions.get_batch(start, end)
        # Give them to online LDA
        (gamma, bound) = olda.update_lambda(docset)
        allgamma[start:end,:] = gamma
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))
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
    vocab = file('./vocab2.txt').readlines()
    W = len(vocab)
    # the data
    data = cu.get_sample_data_frame(datasize)
    test = cu.get_test_data_frame(testsize)
    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
    make_topic_columns(olda, data, K, batchsize)
    make_topic_columns(olda, test, K, batchsize)
        
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
