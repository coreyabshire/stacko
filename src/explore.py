import os
import re
import csv

def word_frequency(corpus):
    words = {}
    for q in corpus:
        for w in q.split():
            if w in words:
                words[w] += 1
            else:
                words[w] = 1
    return words

def word_freq2(corpus):
    words = {}
    pattern = re.compile('[ ,.$?!@#"$%^&*()\n\t]')
    for q in corpus:
        for block in q.split('\n\n'):
            for line in block:
                if not(line.startswith('\t') or line.startswith('    ')):
                    for word in pattern.split(line.lower()):
                        if word in words:
                            words[word] += 1
                        else:
                            words[word] = 1
    return words

def write_freq(freq, filename, limit=1):
    f = open(filename, 'w')
    for w in freq:
        if freq[w] > limit:
            f.write('"%s", %d\n' % (w, freq[w]))
    f.close()

def write_frequency(file_name, freq, limit=1):
    rows = []
    for w in freq:
        if freq[w] > limit:
            rows.append([w.replace('"','').replace(',',''), freq[w]])
    writer = csv.writer(open(os.path.join(data_path, file_name), "w"),
                        lineterminator="\n")
    writer.writerows(rows)

