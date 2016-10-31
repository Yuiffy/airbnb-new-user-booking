import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
import xgboost as xgb
import time
from datetime import datetime as dt
import operator

TRAIN_FINAL_CSV = 'training_features.csv'
TEST_FINAL_CSV = 'testing_features.csv'
LABEL_FINAL_CSV = 'labels.csv'
OUT_FOLDER = '../outputs/'
MODEL_FOLDER = '../models/'
MAX_DEPTH = 4
ETA = 0.1
NUM_ROUND = 2000
SUB_SAMPLE = 1.0
COL_SAMPLE = 0.7
EARLY_STOP = 10
COL_SAMPLE_SEED = 0
XGBOOST_SEED = 0

param = {'max_depth': MAX_DEPTH, 'eta': ETA, 'silent': 1, 'objective': 'multi:softprob', 'subsample': SUB_SAMPLE,
         'colsample_bytree': 1.0, 'num_class': 12, 'seed': XGBOOST_SEED}


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
    return 'ndcg5', ndcg


def p_ndcg5(preds, dtrain):
    name, q = ndcg5(preds, dtrain)
    return 'p_ndcg5', q * 100


startTime = dt.now()
le = LabelEncoder()
print "start read CSV"
destination = pd.read_csv(LABEL_FINAL_CSV, header=0, index_col=0)
x = pd.read_csv(TRAIN_FINAL_CSV, header=0, index_col=0)
x_test = pd.read_csv(TEST_FINAL_CSV, header=0, index_col=0)
print "Over read Csv"
y = le.fit_transform(destination['country_destination'].values)
idSave = x_test.index

x_size = x.shape[0]
x_all = pd.concat([x, x_test], axis=0)
print x_all.columns, x_all.shape

num_round = NUM_ROUND
cv_rounds = 0
col_sample_seed_now = COL_SAMPLE_SEED - 1
cv_result = []
while cv_rounds < 30:
    col_sample_seed_now += 1

    x_all_2 = x_all.sample(frac=COL_SAMPLE, random_state=col_sample_seed_now, axis=1)
    print x_all_2.columns, x_all_2.shape
    x_2 = x_all_2[:x_size]
    x_test_2 = x_all_2[x_size:]

    dtrain = xgb.DMatrix(x_2, label=y, missing=float('NaN'))
    dtest = xgb.DMatrix(x_test_2, missing=float('NaN'))
    print "startCV!"
    cv_result = xgb.cv(param, dtrain, num_round, nfold=5, feval=ndcg5, seed=XGBOOST_SEED,
                       callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                                  xgb.callback.early_stop(EARLY_STOP)])
    print col_sample_seed_now, cv_result
    cv_rounds = max(1, len(cv_result))
cvndcg = cv_result.iloc[-1]['train-ndcg5-mean']

num_round = cv_rounds
evallist = [(dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, evallist, feval=ndcg5, early_stopping_rounds=EARLY_STOP)

paramCheck = '(' + str(MAX_DEPTH) + ',' + str(ETA) + ',' + str(num_round) + ',' + str(SUB_SAMPLE) + ',' + str(
    COL_SAMPLE) + ')'
timeStr = str(time.strftime('%Y-%m-%d %H%M%S', time.localtime()))
bst.save_model(
    MODEL_FOLDER + 'model' + timeStr + str(cvndcg) + paramCheck + str(param['seed']) + '.model')
bst.dump_model(MODEL_FOLDER + timeStr + "dump.raw.txt")
fscore = bst.get_fscore()
sorted_fscore = sorted(fscore.items(), key=operator.itemgetter(1), reverse=True)
print fscore
print sorted_fscore
ypred = bst.predict(dtest)
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
stTimeStr = '(' + str(startTime)[11:19] + 'start)'
timeSpend = (dt.now() - startTime)

print timeSpend, stTimeStr
sub.to_csv(OUT_FOLDER + 'sub' + timeStr + paramCheck + '.csv',
           index=False)
