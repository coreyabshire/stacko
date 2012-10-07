import webbrowser
from random import Random
from features import Body

nb = NaiveBayesClassifier()

if 'nb_words' in dir() and 'nb_labels' in dir():
    print 'reusing words and labels from cache for nb'
    nb.words = nb_words
    nb.labels = nb_labels
else:
    print 'retraining nb and caching words and labels'
    nb.train_from_texts_by_label(texts)
    nb_words = nb.words
    nb_labels = nb.labels

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
    

def show_n_urls(n=100):
    count = 0
    i = 0
    while count < n and i < len(test.BodyMarkdown):
        urls = get_urls(test.BodyMarkdown[i])
        if len(urls) > 0:
            print count, i, len(urls)
            count += 1
        i += 1

def scan_n_bodies(data, n=100):
    count = 0
    i = 0
    while count < n and i < len(data):
        text = data.BodyMarkdown[i]
        body = Body(text)
        print '%4d %4d %8d %1s %3d %2d %3d %3d %1d' % (
            count,
            i,
            len(text),
            body.has_urls() and 'T' or 'F',
            len(body.get_text_urls()),
            body.count_code_blocks(),
            body.count_lines_of_code(),
            len(body.get_unique_words()),
            labels.index(data.OpenStatus[i])
            )
        count += 1
        i += 1

def browse(data, fea, i):
    """Prints basic info and pulls up the page for a training instance.

    Parameters
    ----------

    data : pandas.DataFrame
        The data frame containing the instance data as loaded
        from pandas using the competition utilities.
    fea : pandas.DataFrame
        The feature set extracted from the dataset using the
        benchmark code, or whatever else was used to train
        the RandomForest classifier from scikit-learn.
    i : int
        The index of the instance to display. This index must
        pull up the same instance across both data and fea.
        
    Returns
    -------
    Nothing
    
    Notes
    -----
    This function is used mostly to browse the instances in the training
    set, and gain intuition about what aspects of a question really cause
    it to be marked as closed, and for what reason. Seeing the question
    as it is displayed in the browser helps us gain the perspective of
    the community moderator who would have been viewing it. We can also
    see the latest statistics for the owner of the question, the moderators,
    the context of the closure, and other things not provided in the
    training set. Seeing this alongside the raw data and derived statistics
    and automated classifications is helpful to gain insight into what our
    model is currently thinking as compared to the humans. Comparing these
    two and assessing gaps is helpful in improving the model so that it
    becomes more accurate in its predictions.
    """
    pid = data.PostId[i]
    print data.BodyMarkdown[i]
    print 'Data Index:    %s' % i
    print 'PostId:        %s' % data.PostId[i]
    print 'PostCreation:  %s' % data.PostCreationDate[i]
    print 'OwnerUserId:   %s' % data.OwnerUserId[i]
    print 'OwnerCreation: %s' % data.OwnerCreationDate[i]
    print 'Title:         %s' % data.Title[i]
    print 'Tags:          %s' % [data['Tag%s'%t][i] for t in range(1,6) if not pd.isnull(data['Tag%s'%t][i])]
    print 'Reputation:    %s' % data.ReputationAtPostCreation[i]
    print 'Answers:       %s' % data.OwnerUndeletedAnswerCountAtPostTime[i]
    print 'Title Length:  %s' % len(data.Title[i])
    print 'Body Length:   %s' % len(data.BodyMarkdown[i])
    print 'Num Tags:      %s' % (5 - sum(1 for t in range(1,6) if pd.isnull(data['Tag%s'%t][i])))
    print 'Owner Age:     %s' % (data.PostCreationDate[i] - data.OwnerCreationDate[i])
    print 'OpenStatus:    %s' % data.OpenStatus[i]
    print 'Naive Bayes:   %s' % [round(x,4)*100 for x in normalize_probs(nb.predict(data.BodyMarkdown[74894], sorted_labels))]
    #print 'Random Forest: %s' % [round(x,4)*100 for x in rf.predict_proba(fea.ix[i])[0]]
    url = 'http://stackoverflow.com/questions/%s/' % pid
    return webbrowser.open(url)

def browse_random(data, fea):
    browse(data, fea, random.randint(0, len(data)))

def test_it(data, test):
    c1 = Counter(data[test].OpenStatus)
    return c1, c2
