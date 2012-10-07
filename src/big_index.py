import competition_utilities
import csv
import dateutil
import random
import nltk
import features
from collections import defaultdict

def get_words(raw):
    pass

if __name__ == '__main__':
    count = 0
    #db.train.remove({})
    master = defaultdict(int)
    master_status = defaultdict(lambda: defaultdict(int))
    status_counts = defaultdict(int)
    with open('../data/train.csv', 'rb') as infile:
        reader = csv.reader(infile)
        header = reader.next()
        print header
        body_index = header.index('BodyMarkdown')
        status_index = header.index('OpenStatus')
        print 'Body index is at ', body_index
        print 'Status index is at ', status_index
        for line in reader:
            body = features.Body(line[body_index])
            status = line[status_index]
            words = body.get_unique_words()
            for word in words:
                master[word] += 1
                master_status[status][word] += 1
            if (count % 1000) == 0:
                print count
            if (count > 5000000):
                break
            count += 1
            status_counts[status] += 1
        print count
    print 'writing outfile'
    labels = sorted(status_counts.keys())
    with open('../data2/master.csv', 'wb') as outfile:
        writer = csv.writer(outfile)
        header = ['word']
        for label in labels:
            header.append(label)
        header.append('total')
        for label in labels:
            header.append('p(%s)' % label)
        header.append('p(total)')
        for label in labels:
            header.append('p2(%s)' % label)
        writer.writerow(header)
        for word in sorted(master.keys()):
            if master[word] <= 1:
                continue
            row = [word]
            for label in labels:
                row.append(master_status[label][word])
            row.append(master[word])
            for label in labels:
                row.append(float(master_status[label][word]) / status_counts[label])
            row.append(float(master[word]) / float(count))
            for label in labels:
                row.append(float(master_status[label][word]) / master[word])
            writer.writerow(row)
    print '\n'.join('%20s %9d' % (k,v) for k,v in status_counts.items())
        
