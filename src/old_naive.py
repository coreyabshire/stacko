class NaiveBayesClassifier():
    '''Classifies instances according to word probabilities.'''

    def __init__(self):
        self.words = defaultdict(lambda: defaultdict(int))
        self.labels = defaultdict(int)
        
    def train(self, words, label):
        self.labels[label] += 1;
        for word in words:
            self.words[word][label] += 1
    
    def train_from_texts_by_label(self, texts_by_label):
        for label in texts_by_label.keys():
            for text in texts_by_label[label]:
                body = Body(text)
                self.train(body.get_unique_words(), label)

    def probability(self, text):
        return 0.0

    def word_probability(self, word, label): # p(a|b)
        return float(self.words[word][label]) / float(self.labels[label])

    def weighted_probability(self, word, label):
        pword = self.word_probability(word, label)
        total = sum([self.words[word][label] for label in self.labels.keys()])
        return (0.5 + (total * pword)) / (1 + total)

    def text_probability(self, words, label):
        p = 1.0
        for word in words:
            pword = self.weighted_probability(word, label)
            # print "%s - %s - %s" % (word, label, pword)
            p *= pword
        return p

    def probability(self, words, label):
        plabel = float(self.labels[label]) / float(sum(self.labels.values()))
        pwords = self.text_probability(words, label)
        return  pwords * plabel
        
    def predict(self, text, labels):
        body = Body(text)
        words = body.get_unique_words()
        return [self.probability(words, label) for label in labels]

