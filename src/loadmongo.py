import pymongo
import competition_utilities
import csv
import dateutil
import random

def cleanup_rec(r):
    for k,v in r.items():
        if type(v) == np.int64:
            r[k] = int(v)
        if type(v) == float and math.isnan(v):
            r[k] = None
    return r

def save_to_mongo(db, data):
    for i in xrange(len(data)):
        db.train.save(cleanup_rec(data.ix[i].to_dict()))

if __name__ == '__main__':
    connection = pymongo.Connection()
    db = connection['stacko']
    count = 0
    jackpot = random.randint(1, 1000)
    #db.train.remove({})
    with open('../data2/train-sample.csv', 'rb') as infile:
        reader = csv.reader(infile)
        header = reader.next()
        print header
        for line in reader:
            row = dict(zip(header, line))
            for k,v in row.items():
                if v == '':
                    row[k] = None
                elif k.endswith('Date'):
                    if v != 'nan':
                        try:
                            row[k] = dateutil.parser.parse(v)
                        except ValueError:
                            print 'Date error:', v
                            row[k] = None
                    else:
                        row[k] = None
                elif k.startswith('Rep') or k.find('Count') > 0:
                    row[k] = int(v)
                elif k.endswith('Id'):
                    row[k] = int(v)
            if jackpot == 0:
                print row
                jackpot = random.randint(1, 1000)
            else:
                jackpot -= 1
            if (count % 1000) == 0:
                print count
            db.sample2.save(row)
            count += 1
        print count
