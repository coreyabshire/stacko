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
    maxdocs = 100
    with open('../data/train-sample.csv', 'rb') as infile:
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
            if (count > maxdocs):
                break
            count += 1
            status_counts[status] += 1
        print count
    vocab = [k for k,v in master.items() if v >= 2]
    vocab.sort()
    print 'writing outfile'
    labels = sorted(status_counts.keys())
    count2 = 0
    with open('../data2/tfidf.csv', 'wb') as outfile:
        writer = csv.writer(outfile)
        out_header = ['postid', 'status']
        for word in vocab:
            out_header.append(word)
        writer.writerow(out_header)
        with open('../data/train-sample.csv', 'rb') as infile:
            reader = csv.reader(infile)
            header = reader.next()
            print header
            postid_index = header.index('PostId')
            body_index = header.index('BodyMarkdown')
            status_index = header.index('OpenStatus')
            print 'PostId index is at ', postid_index
            print 'Body index is at ', body_index
            print 'Status index is at ', status_index
            for line in reader:
                row = []
                postid = line[postid_index]
                body = features.Body(line[body_index])
                status = line[status_index]
                words = body.get_unique_words()
                if (count2 > maxdocs):
                    break
                row.append(postid)
                row.append(status)
                for word in vocab:
                    row.append(word in words and 1 or 0)
                writer.writerow(row)
                count2 += 1
    print '\n'.join('%20s %9d' % (k,v) for k,v in status_counts.items())
        
