import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
import xgboost as xgb
import time
from datetime import datetime as dt
import operator

OUT_FOLDER = '../outputs/'
MODEL_FOLDER = '../models/'
TRAIN_FINAL_CSV = 'training_features.csv'
TEST_FINAL_CSV = 'testing_features.csv'
LABEL_FINAL_CSV = 'labels.csv'
BLEED_SAMPLE = 0.8
HOLD_OUT_SEED = 0

# weight, ntree_limit, fname
# MODELS = (
#     (162, '2016-06-26 055219 0.834276(8,0.05,162,0.8,0.8)-1.model'),
#     (72, '2016-06-26 072327 0.834133(8,0.05,72,0.8,0.8)-1.model'),
#     (48, '2016-06-26 080631 0.833844(8,0.05,48,0.8,0.8)-1.model'),
#     (99, '2016-06-26 091507 0.834227(8,0.05,99,0.8,0.8)-1.model'),
#     (94, '2016-06-26 102322 0.833818(8,0.05,94,0.8,0.8)-1.model'),
#     (81, '2016-06-26 112445 0.833842(8,0.05,81,0.8,0.8)-1.model'),
#     (19, '2016-06-26 224036 0.833122(8,0.05,19,0.9,0.9)103(0.9,0).model'),
#     (15, '2016-06-26 230849 0.833265(8,0.05,15,0.9,0.9)104(0.9,0).model'),
#     (149, '2016-06-27 004537 0.834365(8,0.05,149,0.9,0.9)105(0.9,0).model'),
# )
MODELS = (
    (51, '2016-06-27 114538 0.833471(8,0.05,51,0.95,0.9)114(0.8,0).model'),
    (137, '2016-06-27 111952 0.833921(8,0.05,137,0.95,0.9)113(0.8,0).model'),
    (107, '2016-06-27 102312 0.833984(8,0.05,107,0.95,0.9)112(0.8,0).model'),
    (38, '2016-06-27 093724 0.833344(8,0.05,38,0.95,0.9)111(0.8,0).model'),
    (101, '2016-06-27 091620 0.834001(8,0.05,101,0.95,0.9)110(0.8,0).model'),
    (141, '2016-06-27 083233 0.834157(8,0.05,141,0.95,0.9)109(0.8,0).model'),
    (47, '2016-06-27 073433 0.833156(8,0.05,47,0.95,0.9)108(0.8,0).model'),
    (54, '2016-06-27 071013 0.833283(8,0.05,54,0.95,0.9)107(0.8,0).model'),
    (132, '2016-06-27 064320 0.834023(8,0.05,132,0.95,0.9)106(0.8,0).model'),
    (117, '2016-06-27 054837 0.83391(8,0.05,117,0.95,0.9)105(0.8,0).model'),
    (156, '2016-06-27 045904 0.834277(8,0.05,156,0.95,0.9)104(0.8,0).model'),
)

param2 = {'max_depth': 6, 'eta': 0.01, 'silent': 1, 'objective': 'multi:softprob', 'subsample': 0.9,
          'colsample_bytree': 0.9, 'num_class': 12, 'seed': 0}
NUM_ROUND2 = 2000
EARLY_STOP2 = 30

flags = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']
scores = [1.0, 0.63092975357145753, 0.5, 0.43067655807339306, 0.38685280723454163]
other_score = 0.0


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
le.fit(flags)
print le.classes_
print "start read CSV"
destination = pd.read_csv(LABEL_FINAL_CSV, header=0, index_col=0)
x = pd.read_csv(TRAIN_FINAL_CSV, header=0, index_col=0)
x_test = pd.read_csv(TEST_FINAL_CSV, header=0, index_col=0)
print "Over read Csv"
y = le.fit_transform(destination['country_destination'].values)
idSave = x_test.index
x_flg = pd.concat([x, destination], axis=1)
x_in = x_flg.sample(frac=BLEED_SAMPLE, random_state=HOLD_OUT_SEED, axis=0)
x_out = x_flg.drop(x_in.index, axis=0)
y_in = le.transform(x_in['country_destination'].values)
y_out = le.transform(x_out['country_destination'].values)
x_in.drop('country_destination', axis=1, inplace=True)
x_out.drop('country_destination', axis=1, inplace=True)
print x_in.shape, x_out.shape, len(y_in), len(y_out)
dtrain_in = xgb.DMatrix(x_in, label=y_in, missing=float('NaN'))
dtrain_out = xgb.DMatrix(x_out, label=y_out, missing=float('NaN'))
dtest = xgb.DMatrix(x_test, missing=float('NaN'))

print "start read models"
outpreds = np.array
ypred1s = np.array
outarray = []
predarray = []
for one in MODELS:
    tempbst = xgb.Booster(model_file=MODEL_FOLDER + one[1])
    outarray += [tempbst.predict(dtrain_out, ntree_limit=one[0])]
    predarray += [tempbst.predict(dtest, ntree_limit=one[0])]
outpreds = np.concatenate(outarray, axis=1)
ypred1s = np.concatenate(predarray, axis=1)
predcolname = map(lambda x: 'pred' + str(int(x / 12)) + '_' + str(x % 12), range(outpreds.shape[1]))
outdf = pd.DataFrame(outpreds, columns=predcolname, index=x_out.index)
pred1df = pd.DataFrame(ypred1s, columns=predcolname, index=x_test.index)
# traindf2 = pd.concat([x_out, outdf], axis=1)
# testdf2 = pd.concat([x_test, pred1df], axis=1)
traindf2 = outdf
testdf2 = pred1df

# what = pd.DataFrame([], columns=predcolname, index=x_in.index)
# what = pd.concat([x_in, what], axis=1)
# traindf2 = pd.concat([what, traindf2], axis=0)
# traindf2 = pd.concat([traindf2, destination], axis=1)
# y_what = le.transform(traindf2['country_destination'].values)
# traindf2.drop('country_destination', axis=1, inplace=True)

print traindf2.shape, testdf2.shape
outdtrain = xgb.DMatrix(traindf2, label=y_out, missing=float('NaN'))
pred1dtrain = xgb.DMatrix(testdf2, missing=float('NaN'))
print "over read models"

cv_result = xgb.cv(param2, outdtrain, NUM_ROUND2, nfold=5, feval=ndcg5, seed=0,
                   callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                              xgb.callback.early_stop(EARLY_STOP2)])
print cv_result
cv_rounds = max(1, len(cv_result))
cvndcg = cv_result.iloc[-1]['train-ndcg5-mean']
NUM_ROUND2 = cv_rounds

evallist2 = [(outdtrain, 'outdtrain')]
evals_result = {}
bst2 = xgb.train(param2, outdtrain, NUM_ROUND2, evallist2, feval=ndcg5,
                 evals_result=evals_result)
fscore = bst2.get_fscore()
sorted_fscore = sorted(fscore.items(), key=operator.itemgetter(1), reverse=True)
print sorted_fscore

ypred = bst2.predict(pred1dtrain)

# Taking the 5 classes with highest probabilities
ids = []  # list of ids
cts = []  # list of countries
for i in range(len(ypred)):
    idx = idSave[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(ypred[i])[::-1])[:5].tolist()
timeStr = str(time.strftime('%Y-%m-%d %H%M%S', time.localtime()))
the_rounds = map(lambda y: y[0], MODELS)
model_name = timeStr + " bleed " + str(len(MODELS)) + str(the_rounds) + '(' + str(cvndcg) + ')'
# Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv(OUT_FOLDER + 'sub' + model_name + '.csv',
           index=False)  # bst.dump_model(MODEL_FOLDER + model_name + "dump.raw.txt")

stTimeStr = '(' + str(startTime)[11:19] + 'start)'
timeSpend = (dt.now() - startTime)
print timeSpend, stTimeStr
