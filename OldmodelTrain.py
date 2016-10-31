import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
import xgboost as xgb
import time
from datetime import datetime as dt
import matplotlib.pyplot as plt

TRAIN_FINAL_CSV = 'training_features.csv'
TEST_FINAL_CSV = 'testing_features.csv'
LABEL_FINAL_CSV = 'labels.csv'
OUT_FOLDER = '../outputs/'
MAX_DEPTH = 2
ETA = 0.3
NUM_ROUND = 3
SUB_SAMPLE = 1.0
COLSAMPLE_BYTREE = 1.0


# ndcg5
def ndcg5(preds, dtrain):
    k = 5
    y_true = dtrain.get_label()
    n = len(y_true)
    num_class = preds.shape[1]
    index = np.argsort(preds, axis=1)
    top = index[:, -k:][:, ::-1]
    rel = (np.reshape(y_true, (n, 1)) == top).astype(int)
    cal_dcg = lambda y: sum((2 ** y - 1) / np.log2(range(2, k + 2)))
    ndcg = np.mean((np.apply_along_axis(cal_dcg, 1, rel)))
    return 'ndcg5', -ndcg


startTime = dt.now()
# idSave = testCsv['id']
le = LabelEncoder()
destination = pd.read_csv(LABEL_FINAL_CSV, header=0, index_col=0)
# print destination
y = le.fit_transform(destination['country_destination'].values)
x = pd.read_csv(TRAIN_FINAL_CSV, header=0, index_col=0)
x_test = pd.read_csv(TEST_FINAL_CSV, header=0, index_col=0)
idSave = x_test.index

# data = np.random.rand(100000,100)
# label = np.array([1 if int(sum(x)/10)%2==0 else 0 for x in data])
# x = data
# y = label
# sz = x.size
# all = pd.concat((x,x_test),axis=0)
# x = all[:sz]
# x_test = all[sz:]
# x.drop(['secs_elapsed','actions_num'], axis=1, inplace=True)
# x_test.drop(['secs_elapsed','actions_num'], axis=1, inplace=True)
# print x, x_test
# x.columns = [str(i) for i in range(len(x.columns))]
print x.columns
# x.columns = map(lambda q: q.translate(None, '_'), x.columns)
# x.columns = map(lambda q, w: q[:4] + q[len(q) - 2:len(q) + 2] + q[-4:] + str(w) if len(q) > 15 else q, x.columns,
#                 range(len(x.columns)))
# print x.columns
# x_test.columns = x.columns
# x_test.columns = [str(i) for i in range(len(x_test.columns))]
dtrain = xgb.DMatrix(x, label=y, missing=float('NaN'))
dtest = xgb.DMatrix(x_test, missing=float('NaN'))
param = {'max_depth': MAX_DEPTH, 'eta': ETA, 'silent': 1, 'objective': 'multi:softprob', 'subsample': SUB_SAMPLE,
         'colsample_bytree': COLSAMPLE_BYTREE, 'num_class': 12}
# param['nthread']=1
# param['eval_metric'] = 'mlogloss'
# param['eval_metric']=['auc','ams@0']
evallist = [(dtrain, 'train')]
num_round = NUM_ROUND
# #evallist,
# print 'wow!'
# cvcv = xgb.cv(param, dtrain, 1, nfold=3, feval=ndcg5, early_stopping_rounds=30 ,seed=0)
# print cvcv
bst = xgb.train(param, dtrain, num_round, evallist, feval=ndcg5, early_stopping_rounds=30)
ypred = bst.predict(dtest)
# xgb.plot_importance(bst)
print bst.get_dump()
print len(bst.get_dump())
for i in range(len(bst.get_dump())):
    xgb.plot_tree(bst, num_trees=i)
plt.show()

# print ypred[:5]
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

print timeSpend, stTimeStr
sub.to_csv(OUT_FOLDER + 'sub' + time.strftime('%Y-%m-%d %H%M%S', time.localtime()) + paramCheck + '.csv',
           index=False)
