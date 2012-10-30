from __future__ import division
from collections import Counter
import csv
import dateutil
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import pandas as pd
import pymongo

data_path = "C:/Projects/ML/stacko/data2"
submissions_path = data_path
if not data_path or not submissions_path:
    raise Exception("Set the data and submission paths in competition_utilities.py!")

def parse_date_maybe_null(date):
    if date:
        return dateutil.parser.parse(date)
    return None

df_converters = {"PostCreationDate": dateutil.parser.parse,
                 "OwnerCreationDate": dateutil.parser.parse}
                # "PostClosedDate": parse_date_maybe_null}

# It may be more convenient to get rid of the labels and just use
# numeric identifiers for the categories. I would want to do this
# in the order given below, which models the relative frequency of
# each category in the training set provided. This has the added
# convenience that open vs closed becomes 0 vs not 0. I can always
# look up the text labels later in the array given below.

labels = [
    'open',
    'not a real question',
    'off topic',
    'not constructive',
    'too localized']

sorted_labels = [label for label in labels]
sorted_labels.sort()

def get_reader(file_name="train-sample.csv"):
    reader = csv.reader(open(os.path.join(data_path, file_name)))
    header = reader.next()
    return reader

def get_header(file_name="train-sample.csv"):
    reader = csv.reader(open(os.path.join(data_path, file_name)))
    header = reader.next()
    return header

def get_closed_count(file_name):
    return sum(1 for q in iter_closed_questions(file_name))

def iter_closed_questions(file_name):
    df_iter = pd.io.parsers.read_csv(os.path.join(data_path, file_name), iterator=True, chunksize=1000)
    return (question[1] for df in df_iter for question in df[df["OpenStatus"] != "open"].iterrows())

def iter_open_questions(file_name):
    df_iter = pd.io.parsers.read_csv(os.path.join(data_path, file_name), iterator=True, chunksize=1000)
    return (question[1] for df in df_iter for question in df[df["OpenStatus"] == "open"].iterrows())

def get_dataframe(file_name="train-sample.csv"):
    return pd.io.parsers.read_csv(os.path.join(data_path, file_name), converters = df_converters)

def compute_priors(closed_reasons):
    closed_reason_counts = Counter(closed_reasons)
    reasons = sorted(closed_reason_counts.keys())
    total = len(closed_reasons)
    priors = [closed_reason_counts[reason]/total for reason in reasons]
    return priors

def get_priors(file_name):
    return compute_priors([r[14] for r in get_reader(file_name)])

def write_sample(file_name, header, sample):
    writer = csv.writer(open(os.path.join(data_path, file_name), "w"), lineterminator="\n")
    writer.writerow(header)
    writer.writerows(sample)

def update_prior(old_prior,  old_posterior, new_prior):
    evidence_ratio = (old_prior*(1-old_posterior)) / (old_posterior*(1-old_prior))
    new_posterior = new_prior / (new_prior + (1-new_prior)*evidence_ratio)
    return new_posterior

def cap_and_update_priors(old_priors, old_posteriors, new_priors, epsilon):
    old_posteriors = cap_predictions(old_posteriors, epsilon)
    old_priors = np.kron(np.ones((np.size(old_posteriors, 0), 1)), old_priors)
    new_priors = np.kron(np.ones((np.size(old_posteriors, 0), 1)), new_priors)
    evidence_ratio = (old_priors*(1-old_posteriors)) / (old_posteriors*(1-old_priors))
    new_posteriors = new_priors / (new_priors + (1-new_priors)*evidence_ratio)
    new_posteriors = cap_predictions(new_posteriors, epsilon)
    return new_posteriors

def cap_predictions(probs, epsilon):
    probs[probs>1-epsilon] = 1-epsilon
    probs[probs<epsilon] = epsilon
    row_sums = probs.sum(axis=1)
    probs = probs / row_sums[:, np.newaxis]
    return probs

def write_submission(file_name, predictions):
    with open(os.path.join(submissions_path, file_name), "w") as outfile:
        writer = csv.writer(outfile, lineterminator="\n")
        writer.writerows(predictions)

def get_submission_reader(file_name="submission.csv"):
    reader = csv.reader(open(os.path.join(data_path, file_name)))
    return reader

def get_dataframe_mongo_query(query):
    db = pymongo.Connection().stacko
    return pd.DataFrame([r for r in db.train.find(query)])

def get_sample_data_frame(limit=0):
    db = pymongo.Connection().stacko
    return pd.DataFrame([r for r in db.sample2.find(limit=limit)])

def get_sample_data_frame_by_status(status, limit=0):
    db = pymongo.Connection().stacko
    return pd.DataFrame([r for r in db.sample2.find(
                {"OpenStatus": status}, limit=limit)])

def get_test_data_frame(limit=0):
    db = pymongo.Connection().stacko
    return pd.DataFrame([r for r in db.test2.find(limit=limit)])

def get_dataframe_mongo_date_range(start, end):
    query = {'PostCreationDate': {'$gte': start, '$lt': end}}
    return get_dataframe_mongo_query(query)

def get_dataframe_mongo_months(year, month, count):
    start = datetime(year, month, 1)
    relative = relativedelta(months = count)
    end = start + relative
    return get_dataframe_mongo_date_range(start, end)

def load_priors(name='train.csv'):
    return pymongo.Connection().stacko.priors.find_one({'for':name})['priors']
