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
# weight, ntree_limit, fname
SUBMITS = ((5, 0.88579, 'sub2016-06-19 021351 0.123456(8,0.05,143,0.8,0.8).csv', 0.834143),
           (4.9, 0.88577, 'sub2016-06-10 000430(8,0.05,120,0.8,0.8).csv'),
           (2, 0.88541, 'sub2016-06-19 075827 0.833886(8,0.01,616,0.8,0.8).csv'),
           (2.1, 0.88544, 'sub2016-06-19 092129 0.833746(6,0.01,1233,0.8,0.8).csv'),
           (1.9, 0.88535, 'sub2016-06-18 223721 0.123456(8,0.05,120,0.8,0.8).csv'),
           (1, 0.88489, 'sub2016-06-19 003559 0.834143(8,0.05,144,0.8,0.8).csv'),
           (1, 0.88488, 'sub2016-06-19 171059 0.833726(8,0.05,123,0.8,0.8).csv'),
           # (5, 0.00000, 'sub2016-06-21 125038 0.834295(10,0.05,109,0.9,0.9)2.csv'),
           # (5, 0.00000, 'sub2016-06-21 144620 0.834176(10,0.05,104,0.9,0.9)5.csv'),
           # (5, 0.00000, 'sub2016-06-21 114638 0.833316(10,0.05,45,0.9,0.9)1.csv'),
           # (5, 0.00000, 'sub2016-06-21 134504 0.833653(10,0.05,46,0.9,0.9)4.csv'),
           # (5, 0.00000, 'sub2016-06-21 152108 0.83355(10,0.05,54,0.9,0.9)6.csv'),
           # (5, 0.00000, 'sub2016-06-21 154706 0.833431(10,0.05,38,0.9,0.9)7.csv'),
           # (5, 0.00000, 'sub2016-06-21 161824 0.833223(10,0.05,49,0.9,0.9)8.csv'),
           # (5, 0.00000, 'sub2016-06-21 165435 0.833683(10,0.05,58,0.9,0.9)9.csv'),
           # (5, 0.00000, 'sub2016-06-21 174011 0.833385(10,0.05,46,0.9,0.9)11.csv'),
           # (5, 0.00000, 'sub2016-06-21 184529 0.833422(10,0.05,47,0.9,0.9)14.csv'),
           # (5, 0.00000, 'sub2016-06-21 191637 0.833005(10,0.05,48,0.9,0.9)15.csv')
           )
flags = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']
scores = [1.0, 0.63092975357145753, 0.5, 0.43067655807339306, 0.38685280723454163]
other_score = 0.0

startTime = dt.now()
le = LabelEncoder()
le.fit(flags)
print le.classes_
print "start read CSV"
csvs = [0] * len(SUBMITS)
args = [0] * len(SUBMITS)
for i in range(len(SUBMITS)):
    csvs[i] = pd.read_csv(OUT_FOLDER + SUBMITS[i][2], header=0, index_col=0)
    args[i] = le.transform(csvs[i]['country'])
print args[0]
print "Over read Csv"
row_num = int(csvs[0].size / 5)
idSave = pd.unique(csvs[0].index)
sum = np.array([[0.0] * 12 for i in range(row_num)])
for csv, arg, submit in zip(csvs, args, SUBMITS):
    adds = np.array([[other_score] * 12 for i in range(row_num)])
    j = -5
    for i in xrange(0, row_num):
        j += 5
        adds[i][arg[j:j + 5]] = scores
    sum += adds * submit[0]
print sum[:5]

# Taking the 5 classes with highest probabilities
ids = []  # list of ids
cts = []  # list of countries
for i in range(row_num):
    idx = idSave[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(sum[i])[::-1])[:5].tolist()
timeStr = str(time.strftime('%Y-%m-%d %H%M%S', time.localtime()))
weights = map(lambda y: y[0], SUBMITS)
model_name = timeStr + " bagging " + str(len(SUBMITS)) + str(weights)
# Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv(OUT_FOLDER + 'sub' + model_name + '.csv',
           index=False)  # bst.dump_model(MODEL_FOLDER + model_name + "dump.raw.txt")

stTimeStr = '(' + str(startTime)[11:19] + 'start)'
timeSpend = (dt.now() - startTime)
print timeSpend, stTimeStr
