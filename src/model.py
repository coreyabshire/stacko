import competition_utilities as cu
from features import Body
import features
import nltk
import re
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

train_file = "train-sample.csv"
full_train_file = "train.csv"
test_file = "public_leaderboard.csv"
submission_file = "submission.csv"

def main():
    print("Reading the data")
    data = cu.get_dataframe(train_file)

    print("Extracting features")
    fea = features.extract_features(data)

    print("Training the model")
    rf = RandomForestClassifier(n_estimators=50, verbose=2, compute_importances=True, n_jobs=1)
    rf.fit(fea, data["OpenStatus"])

    print("Reading test file and making predictions")
    data = cu.get_dataframe(test_file)
    test_features = features.extract_features(data)
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
