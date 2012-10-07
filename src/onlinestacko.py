#!/usr/bin/python

# onlinewikipedia.py: Demonstrates the use of online VB for LDA to
# analyze a bunch of random Wikipedia articles.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy
import sys
import onlineldavb
import competition_utilities
import csv

datafilename = 'C:/Projects/ML/stacko/data2/train-sample.csv'

def istag(x):
    return x != 'nan'

class QuestionSet:

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

def main():
    """
    Downloads and analyzes a bunch of random Wikipedia articles using
    online VB for LDA.
    """

    # The number of documents to analyze each iteration
    batchsize = 64
    # The total number of questions on Stack Overflow
    D = 3.3e6
    # The number of topics
    K = 50

    # How many documents to look at
    if (len(sys.argv) < 2):
        documentstoanalyze = int(D/batchsize)
    else:
        documentstoanalyze = int(sys.argv[1])

    # Our vocabulary
    vocab = file('./vocab2.txt').readlines()
    W = len(vocab)

    # Our set of questions from Stack Overflow
    questions = QuestionSet(datafilename)

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    print 'processing', documentstoanalyze
    for iteration in range(0, documentstoanalyze):
        # Download some articles
        (docset, articlenames) = questions.get_batch(batchsize)
        # Give them to online LDA
        (gamma, bound) = olda.update_lambda(docset)
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 10 == 0):
            numpy.savetxt('lambda-%d.dat' % iteration, olda._lambda)
            numpy.savetxt('gamma-%d.dat' % iteration, gamma)

if __name__ == '__main__':
    main()

