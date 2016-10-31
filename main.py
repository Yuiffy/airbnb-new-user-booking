import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
import xgboost as xgb
import time
from datetime import datetime as dt

TRAIN_CSV = '../inputs/train_users_2.csv'
TEST_CSV = '../inputs/test_users.csv'
OUT_FOLDER = '../outputs/'
MAX_DEPTH = 8
ETA = 0.3
NUM_ROUND = 5
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


def toVector(data):
    print 'to!'
    # format:2010-06-28
    createdDates = map(lambda x: map(int, x.split('-')), data['date_account_created'])
    createdYearMonthDay = pd.DataFrame(map(lambda x: {'cYear': x[0], 'cMonth': x[1], 'cDay': x[2]}, createdDates))
    # format:20090319043255
    firstActiveTimestamp = map(lambda x: map(int, [x[:4], x[4:6], x[6:8], x[8:10], x[10:12], x[12:14]]),
                               data['timestamp_first_active'].astype(str))
    firstActiveThings = pd.DataFrame(
        map(lambda x: {'faYear': x[0], 'faMonth': x[1], 'faDay': x[2], 'faHour': x[3], 'faMin': x[4], 'faSec': x[5]},
            firstActiveTimestamp))
    # print createdYearMonthDay
    # One-hot-encoding features
    ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
                 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    df_all = pd.DataFrame()
    for f in ohe_feats:
        df_all_dummy = pd.get_dummies(data[f], prefix=f)
        df_all = pd.concat((df_all, df_all_dummy), axis=1)
    # av = data.age.values
    # data['age'] = np.where(np.logical_or(av<14, av>100), -1, av)
    re = pd.concat((createdYearMonthDay, df_all, firstActiveThings, data['age']), axis=1)
    # print re
    # re = pd.DataFrame([{'x': 1}] * len(re))
    #print re[:1]
    #what = pd.DataFrame(re)[:1]
    #what.to_csv(str(time.localtime())+'.csv')
    return re


trainCsv = pd.read_csv(TRAIN_CSV, header=0)
testCsv = pd.read_csv(TEST_CSV, header=0)

startTime = dt.now()
destination = trainCsv['country_destination'].values
trainCsv = trainCsv.drop('country_destination', axis=1)
idSave = testCsv['id']
pivTrain = trainCsv.shape[0]
#concat train and test, if make vector alternative, the vector will broken, so we must concat them.
dfAll = pd.concat((trainCsv, testCsv), axis=0, ignore_index=True)
dfAllVector = toVector(dfAll)
le = LabelEncoder()

y = le.fit_transform(destination)
x = dfAllVector[:pivTrain]
x_test = dfAllVector[pivTrain:]

TRAIN_FINAL_CSV = 'training_features.csv'
TEST_FINAL_CSV = 'testing_features.csv'
# X = x
# x = pd.read_csv(TRAIN_FINAL_CSV, header=0, index_col=0)
# print x,X
#x_test = pd.read_csv(TEST_FINAL_CSV, header=0, index_col=0)
print x,x_test
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
print ypred[:5]
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
