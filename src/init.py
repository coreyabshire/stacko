data = pd.load('../data2/train-sample.pickle')
texts = features.read_texts_by_label(data)
test = cu.get_dataframe('public_leaderboard.csv')
test_texts = features.read_texts_by_label(test)
