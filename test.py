import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
import xgboost as xgb
import time
from datetime import datetime as dt
import matplotlib.pyplot as plt
import operator

TRAIN_FINAL_CSV = 'training_features.csv'
TEST_FINAL_CSV = 'testing_features.csv'
LABEL_FINAL_CSV = 'labels.csv'
OUT_FOLDER = '../outputs/'
MODEL_FOLDER = '../models/'
fname = '2016-06-20 194514 0.834245(9,0.05,217,1.0,1.0)0.model'
MAX_DEPTH = -123
ETA = 0.3
NUM_ROUND = 1
SUB_SAMPLE = 1.0
COLSAMPLE_BYTREE = 1.0


# ndcg5
def ndcg5(preds, dtrain):
    k = 5
    y_true = dtrain.get_label()
    n = len(y_true)
    num_class = preds.shape[1]
    print "shape:", preds.shape
    index = np.argsort(preds, axis=1)
    top = index[:, -k:][:, ::-1]  # lines[:], columns[-k:] ; then lines[:], columns[::-1]
    rel = (np.reshape(y_true, (n, 1)) == top).astype(int)
    cal_dcg = lambda y: sum((2 ** y - 1) / np.log2(range(2, k + 2)))
    ndcg = np.mean((np.apply_along_axis(func1d=cal_dcg, axis=1, arr=rel)))
    return 'ndcg5', -ndcg


ndcg5scores = ([1] * 5) / np.log2(range(2, 2 + 5))
print list(ndcg5scores)
exit()
startTime = dt.now()
le = LabelEncoder()
destination = pd.read_csv(LABEL_FINAL_CSV, header=0, index_col=0)
y = le.fit_transform(destination['country_destination'].values)
x_test = pd.read_csv(TEST_FINAL_CSV, header=0, index_col=0)
idSave = x_test.index

dtest = xgb.DMatrix(x_test, missing=float('NaN'))
num_round = NUM_ROUND
bst = xgb.Booster(model_file=MODEL_FOLDER + fname)
bst.dump_model('dump.raw.txt')
# dtest = xgb.DMatrix('dtest.buffer')
ypred = bst.predict(dtest, ntree_limit=3)
# xgb.plot_importance(bst)
# print bst.get_dump()

print len(bst.get_dump())
fscore = bst.get_fscore()
sorted_fscore = sorted(fscore.items(), key=operator.itemgetter(1), reverse=True)
print sorted_fscore
# for i in range(len(bst.get_dump())):
#     xgb.plot_tree(bst, num_trees=i)
# plt.show()

print ypred[:5]
print le.classes_
# Taking the 5 classes with highest probabilities
ids = []  # list of ids
cts = []  # list of countries
for i in range(len(ypred)):
    idx = idSave[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(ypred[i])[::-1])[:5].tolist()

# Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
paramCheck = '(' + str(MAX_DEPTH) + ',' + str(ETA) + ',' + str(NUM_ROUND) + ',' + str(SUB_SAMPLE) + ',' + str(
    COLSAMPLE_BYTREE) + ')'
stTimeStr = '(' + str(startTime)[11:19] + 'start)'
timeSpend = (dt.now() - startTime)

print sub[:10]

print timeSpend, stTimeStr
sub.to_csv(OUT_FOLDER + 'sub' + time.strftime('%Y-%m-%d %H%M%S', time.localtime()) + paramCheck + '.csv',
           index=False)
