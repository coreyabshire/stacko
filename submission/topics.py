import competition_utilities as cu
import numpy
import pandas as pd
import onlineldavb

def istag(x):
    return not pd.isnull(x) #(x is not None) and (x != 'nan')

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
            row = self.dataframe.ix[i]
            doc, name = self.parse_doc(row)
            docset.append(doc)
            articlenames.append(name)
        return docset, articlenames

    def close(self):
        self.file.close()

def allocate_topics(lda, data, K, batchsize, D):
    n_iterations = len(data) / batchsize
    questions = QuestionSet(data)
    topics = numpy.zeros((len(data), K))

    # derive topics from data in batches
    for iteration in range(0, n_iterations):
        start = iteration * batchsize
        end = start + batchsize
        (docset, _) = questions.get_batch(start, end)
        (gamma, bound) = lda.update_lambda(docset)
        topics[start:end,:] = gamma
        (wordids, wordcts) = onlineldavb.parse_doc_list(docset, lda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, lda._rhot, numpy.exp(-perwordbound))

    # copy to dataframe
    for k in range(K):
        data['Topic%d'%k] = topics[:,k]

    return topics
