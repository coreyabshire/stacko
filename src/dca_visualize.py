from pylab import *

data = cu.get_dataframe('train-sample.csv')
fea = feature.extract_features(feature_names, data)
r = range(len(fea))
x = log(fea.BodyMarkdown[r])
y = log(fea.ReputationAtPostCreation[r])
open_status_colormap = {
    'open': 'g',
    'not a real question': 'r',
    'off topic': 'b',
    'not constructive': 'c',
    'too localized': 'm'}
c = [open_status_colormap[s] for s in data.OpenStatus[r]]
scatter(x, y, c=c,  alpha=0.01)
show()
