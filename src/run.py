nb = NaiveBayesClassifier()
#nb.train_from_texts_by_label(texts)

nb.words = nb_words
nb.labels = nb_labels

#nb_words = nb.words
#nb_labels = nb.labels

text = test_texts[labels[1]][0]
words = Body(text).get_unique_words()

def normalize_probs(probs):
    prob_sum = sum(p for p in probs)
    if prob_sum:
        return [p / prob_sum for p in probs]
    else:
        return [0.0 for p in probs]

def try_some():
    for label in labels:
        for i in range(10):
            text = test_texts[label][i]
            probs = normalize_probs(nb.predict(text, labels))
            plabel = labels[probs.index(max(probs))]
            print '%s - %s - %s' % (label, plabel, ['%4.2f' % p for p in probs])

def write_bayesian_submission(data):
    status_column = [
        'not a real question',
        'not constructive',
        'off topic',
        'open',
        'too localized']
    file_name = 'bayesian_benchmark.csv'
    count = 0
    with open(os.path.join(submissions_path, file_name), "wb") as outfile:
        for text in data.BodyMarkdown:
            probs = normalize_probs(nb.predict(text, status_column))
            line = ','.join([str(p) for p in probs])
            #print line
            outfile.write('%s\n' % line)
            count += 1
    return count

def measure_accuracy():
    right = 0
    wrong = 0
    total = 0
    falsepos = 0
    for label in labels:
        for i in range(len(test_texts[label])):
            text = test_texts[label][i]
            probs = nb.predict(text, labels)
            plabel = labels[probs.index(max(probs))]
            if label == plabel:
                right += 1
            else:
                wrong += 1
            total += 1
            if label == 'open' and plabel != 'open':
                falsepos += 1
    print right, 'right'
    print wrong, 'wrong'
    print total, 'total'
    print falsepos, 'falsepos'
    print float(right) / float(total), 'correct'
    print float(falsepos) / float(total), 'falsepos'
                
