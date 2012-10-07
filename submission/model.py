import competition_utilities as cu
from topics import allocate_topics
from features import extract_features
from sklearn.ensemble import RandomForestClassifier
import onlineldavb

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
                , "CodeLength"
                , "NumUrls"
                ]

def main():
    # The number of documents to analyze each iteration
    batchsize = 1000

    # The total number of questions on Stack Overflow
    D = 3.3e6

    # The number of topics
    K = 100

    # Make sure the topics are included as features for analysis
    feature_names.extend('Topic%d' % k for k in range(K))

    print("Reading the vocabulary")
    vocab = file('./vocab2.txt').readlines()

    # How many words are in the vocabulary
    W = len(vocab)

    print("Reading the data")
    data = cu.get_dataframe(train_file)

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    lda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)

    print("Allocating the topics")
    allocate_topics(lda, data, K, batchsize, D)

    print("Extracting features")
    fea = extract_features(feature_names, data)

    print("Training the model")
    rf = RandomForestClassifier(n_estimators=50, verbose=2,
                                compute_importances=True, n_jobs=-1)
    rf.fit(fea, data["OpenStatus"])

    print("Reading test file and making predictions")
    data = cu.get_dataframe(test_file)
    allocate_topics(lda, data, K, batchsize, D)
    test_features = extract_features(feature_names, data)
    probs = rf.predict_proba(test_features)

    print("Calculating priors and updating posteriors")
    new_priors = cu.get_priors(full_train_file)
    old_priors = cu.get_priors(train_file)
    probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
    
    print("Saving submission to %s" % submission_file)
    cu.write_submission(submission_file, probs)

if __name__=="__main__":
    main()
