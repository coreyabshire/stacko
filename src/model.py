import competition_utilities as cu
import features
import nltk
import re
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

train_file = "train-sample.csv"
full_train_file = "train.csv"
test_file = "public_leaderboard.csv"
submission_file = "submission.csv"

feature_names = [ "BodyLength"
                , "NumTags"
                , "OwnerUndeletedAnswerCountAtPostTime"
                , "ReputationAtPostCreation"
                , "TitleLength"
                , "UserAge"
                ]

url_pattern = re.compile('(?P<url>https?://[^\s]+)')
punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def is_code_line(line):
    '''Return true if the line of markdown text is a code line.'''
    return line.startswith('    ') or line.startswith('\t')

def classify_line_for_block(line):
    '''Return 'code' or 'text', depending on the type of line given.'''
    if is_code_line(line):
        return 'code'
    else:
        return 'text'

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
        
    def get_all(self, block_type):
        return '\n'.join(v for t,v in self.blocks if t == block_type)
        
    def get_all_text(self):
        return self.get_all('text')
                
    def get_all_code(self):
        return self.get_all('code')
    
    def get_sentences(self):
        sentences = []
        for block in self.get_blocks('text'):
            for sentence in punkt_tokenizer.tokenize(block):
                sentences.append(sentence.strip())
        return sentences
    
    def get_words(self):
        words = []
        for sentence in self.get_sentences():
            for word in nltk.word_tokenize(sentence):
                words.append(word)
        return words
        
    def get_word_freq(self):
        return nltk.FreqDist([w.lower() for w in self.get_words()])

    def get_unique_words(self):
        return [w for w in self.get_word_freq()]
        
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

    def word_probability(self, word, label):
        return float(self.words[word][label]) / float(self.labels[label])

    def weighted_probability(self, word, label):
        pword = self.word_probability(word, label)
        total = sum([self.words[word][label] for label in self.labels.keys()])
        return (1 + (total * pword)) / (1 + total)

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

def main():
    print("Reading the data")
    data = cu.get_dataframe(train_file)

    print("Extracting features")
    fea = features.extract_features(feature_names, data)

    print("Training the model")
    rf = RandomForestClassifier(n_estimators=10, verbose=2, compute_importances=True, n_jobs=-1)
    rf.fit(fea, data["OpenStatus"])

    print("Reading test file and making predictions")
    data = cu.get_dataframe(test_file)
    test_features = features.extract_features(feature_names, data)
    probs = rf.predict_proba(test_features)

    print("Calculating priors and updating posteriors")
    new_priors = cu.get_priors(full_train_file)
    old_priors = cu.get_priors(train_file)
    probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
    
    print("Saving submission to %s" % submission_file)
    cu.write_submission(submission_file, probs)

def is_code_line(line):
    return line.startswith('    ') or line.startswith('\t')

# ' '.join(body_content(df.BodyMarkdown[1:100]))

def body_content(bodies):
    lines = []
    for body in bodies:
        for line in body.split('\n'):
            if line != '' and not is_code_line(line):
                lines.append(line)
    return lines

def get_content(bodies):
    return ' '.join(body_content(bodies))


def extract_text(df, r=range(1000)):
    return nltk.Text(nltk.word_tokenize(get_content(df.BodyMarkdown[r])))


if __name__=="__main__":
    main()
