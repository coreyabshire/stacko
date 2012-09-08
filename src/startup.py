from __future__ import division
from collections import Counter
import csv
import dateutil
import datetime
import math
import nltk
import numpy as np
import os
import pandas as pd
import pylab as pl
import sys

__name__ = 'repl'
execfile('competition_utilities.py')
execfile('model.py')
#data = cu.get_dataframe('train-sample.csv')
#data = cu.get_dataframe('train.csv')
#data = get_data('train-sample.csv')

df_conv2 = {"PostCreationDate": dateutil.parser.parse,
            "OwnerCreationDate": dateutil.parser.parse,
            "BodyMarkdown": len,
            #"PostClosedDate": parse_date_maybe_null,
            "Title": len}

def get_data(file_name="train-sample.csv"):
    tp = pd.io.parsers.read_csv(os.path.join(data_path, file_name), converters = df_conv2, chunksize=1024)
    return pd.concat([chunk for chunk in tp], ignore_index=True)


def count_rows(file_name="train.csv"):
    count = 0
    reader = csv.reader(open(os.path.join(data_path, file_name)))
    header = reader.next()
    for line in reader:
        count += 1
    return count

def make_week_lookup():
    d = datetime.date(2008, 07, 27)
    today = datetime.date.today()
    one_day = datetime.timedelta(days=1)
    week = 0
    lookup = {}
    format = '%Y-%m-%d'
    while d < today:
        label = d.strftime(format)
        for i in range(7):
            lookup[d.strftime(format)] = label
            d += one_day
        week += 1
    return lookup

def fast_split(file_name="train.csv", out_name="train_out3.csv"):
    count = 0
    start_date = '2011-07-03'
    end_date = '2012-07-08'
    #start_date = '2008-07-03'
    #end_date = '2008-09-08'
    week_lookup = make_week_lookup()
    with open(os.path.join(data_path, file_name), 'rb') as infile:
        reader = csv.reader(infile)
        with open(os.path.join(data_path, out_name), 'wb') as outfile:
            writer = csv.writer(outfile)
            header = reader.next()
            writer.writerow(header)
            for line in reader:
                d = line[1] # post creation date
                if (start_date <= d <= end_date):
                    writer.writerow(line)

def fast_small(file_name="train.csv", out_name="small20.csv", n=20):
    count = 0
    week_lookup = make_week_lookup()
    with open(os.path.join(data_path, file_name)) as infile:
        reader = csv.reader(infile)
        with open(os.path.join(data_path, out_name), 'wb') as outfile:
            writer = csv.writer(outfile)
            header = reader.next()
            writer.writerow(header)
            for line in reader:
                if count > n:
                    break
                d = line[1] # post creation date
                writer.writerow(line)
                count += 1
    

def add_date_labels(file_name="train_out.csv", out_name="train_out2.csv"):
    reader = csv.reader(open(os.path.join(data_path, file_name)))
    writer = csv.writer(open(os.path.join(data_path, out_name), 'wt'))
    header = reader.next()
    header.append('Day')
    header.append('Week')
    writer.writerow(header)
    lookup = make_week_lookup()
    for line in reader:
        day_label = line[1][:10]
        week_label = lookup[day_label]
        line.append(day_label)
        line.append(week_label)
        writer.writerow(line)
    
        
def entities(text):
    return nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)), binary=True)
