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
MAX_DEPTH = 9
ETA = 0.05
NUM_ROUND = 5
SUB_SAMPLE = 1.0
COL_SAMPLE = 1.0
EARLY_STOP = 100
COL_SAMPLE_SEED = 0
XGBOOST_SEED = 0
REPEAT = 1
OV_SAMPLE = 1.0

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


startTime = dt.now()
le = LabelEncoder()
print "start read CSV"
destination = pd.read_csv(LABEL_FINAL_CSV, header=0, index_col=0)
x = pd.read_csv(TRAIN_FINAL_CSV, header=0, index_col=0)
x_test = pd.read_csv(TEST_FINAL_CSV, header=0, index_col=0)
print "Over read Csv"
y = le.fit_transform(destination['country_destination'].values)
idSave = x_test.index

HOLD_OUT_SEED = 0
x_flg = pd.concat([x, destination], axis=1)
x_in = x_flg.sample(frac=OV_SAMPLE, random_state=HOLD_OUT_SEED, axis=0)
x_out = x_flg.drop(x_in.index, axis=0)
y_in = le.transform(x_in['country_destination'].values)
y_out = le.transform(x_out['country_destination'].values)
x_in.drop('country_destination', axis=1, inplace=True)
x_out.drop('country_destination', axis=1, inplace=True)
print x_in.shape, x_out.shape, len(y_in), len(y_out)

num_round = NUM_ROUND
col_sample_seed_now = COL_SAMPLE_SEED
MAX_DEPTH -= 1
for i in range(REPEAT):
    MAX_DEPTH += 1
    param = {'max_depth': MAX_DEPTH, 'eta': ETA, 'silent': 1, 'objective': 'multi:softprob', 'subsample': SUB_SAMPLE,
             'colsample_bytree': 1.0, 'num_class': 12, 'seed': XGBOOST_SEED}
    x_in_droped = x_in.sample(frac=1.0 - COL_SAMPLE, random_state=col_sample_seed_now, axis=1)
    x_in_sampled = x_in.copy()
    x_in_sampled[x_in_droped.columns] = np.nan
    x_out_sampled = x_out.copy()
    x_out_sampled[x_in_droped.columns] = np.nan

    dtrain_in = xgb.DMatrix(x_in_sampled, label=y_in, missing=float('NaN'))
    dtrain_out = xgb.DMatrix(x_out_sampled, label=y_out, missing=float('NaN'))
    dtest = xgb.DMatrix(x_test, missing=float('NaN'))
    evallist = [(dtrain_in, 'train_in'),
                # (dtrain_out, 'train_out')
                ]
    print "start hold-out-validation!"
    evals_result = {}
    oneStartTime = dt.now()
    bst = xgb.train(param, dtrain_in, num_round, evallist, feval=ndcg5, early_stopping_rounds=EARLY_STOP,
                    evals_result=evals_result)
    oneTimeSpend = (dt.now() - oneStartTime)
    print oneTimeSpend, oneStartTime
    if hasattr(bst, 'best_score'):
        print col_sample_seed_now, bst.best_score, bst.best_iteration, bst.best_ntree_limit
        train_rounds = bst.best_ntree_limit
        best_score = bst.best_score
    else:
        train_rounds = num_round
        best_score = evals_result['train_out']['ndcg5'][-1]

    paramCheck = '(' + str(MAX_DEPTH) + ',' + str(ETA) + ',' + str(train_rounds) + ',' + str(SUB_SAMPLE) + ',' + str(
        COL_SAMPLE) + ')'
    timeStr = str(time.strftime('%Y-%m-%d %H%M%S', time.localtime()))
    model_name = timeStr + " " + str(best_score) + paramCheck + str(col_sample_seed_now)
    bst.save_model(MODEL_FOLDER + model_name + '.model')
    fscore = bst.get_fscore()
    sorted_fscore = sorted(fscore.items(), key=operator.itemgetter(1), reverse=True)
    print sorted_fscore
    ypred = bst.predict(dtest, ntree_limit=train_rounds)
    # Taking the 5 classes with highest probabilities
    ids = []  # list of ids
    cts = []  # list of countries
    for i in range(len(ypred)):
        idx = idSave[i]
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(ypred[i])[::-1])[:5].tolist()

    # Generate submission
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    sub.to_csv(OUT_FOLDER + 'sub' + model_name + '.csv', index=False)
# bst.dump_model(MODEL_FOLDER + model_name + "dump.raw.txt")


stTimeStr = '(' + str(startTime)[11:19] + 'start)'
timeSpend = (dt.now() - startTime)
print timeSpend, stTimeStr
