import pandas as pd
import numpy as np
from datetime import datetime as dt
import math

TRAIN_CSV = '../inputs/train_users_2.csv'
TEST_CSV = '../inputs/test_users.csv'
SESSION_CSV = '../inputs/sessions.csv'
TRAIN_FINAL_CSV = 'training_features.csv'
TEST_FINAL_CSV = 'testing_features.csv'
LABEL_FINAL_CSV = 'labels.csv'


def get_user_info_vector(data):
    print 'to!'
    print data.columns
    # account created date, format:2010-06-28
    createdDates = map(lambda x: map(int, x.split('-')), data['date_account_created'])
    # createdYearMonthDay = pd.DataFrame(map(lambda x: {'cYear': x[0], 'cMonth': x[1], 'cDay': x[2]}, createdDates))
    # createdYearMonthDay.set_index(data.index, inplace=True)
    the_date = dt(2000, 1, 1)
    created_date_time = map(lambda q: dt.strptime(q, '%Y-%m-%d'), data['date_account_created'])
    created_date_seconds = pd.DataFrame(map(lambda q: {'cTime': (q - the_date).total_seconds()}, created_date_time))
    created_date_seconds.set_index(data.index, inplace=True)

    # first active timestamp, format:20090319043255
    faFormat = "%Y%m%d%H%M%S"
    # firstActiveTimeArray = map(lambda x: map(int, [x[:4], x[4:6], x[6:8], x[8:10], x[10:12], x[12:14]]),
    #                            data['timestamp_first_active'].astype(str))
    # firstActiveThings = pd.DataFrame(
    #     map(lambda x: {'faYear': x[0], 'faMonth': x[1], 'faDay': x[2], 'faHour': x[3], 'faMin': x[4], 'faSec': x[5]},
    #         firstActiveTimeArray))
    # firstActiveThings.set_index(data.index, inplace=True)
    first_active_date_time = map(lambda q: dt.strptime(q, faFormat),
                                 data['timestamp_first_active'].astype(str))
    first_active_seconds = pd.DataFrame(
        map(lambda q: {'faTime': (q - the_date).total_seconds()}, first_active_date_time))
    first_active_seconds.set_index(data.index, inplace=True)

    # date_first_booking, format:2010-06-29
    date_booking_time = map(lambda q: dt.strptime(q, '%Y-%m-%d') if pd.notnull(q) else the_date,
                            data['date_first_booking'])
    date_booking_seconds = pd.DataFrame(map(lambda q: {'cTime': (q - the_date).total_seconds()}, date_booking_time))
    date_booking_seconds.set_index(data.index, inplace=True)

    elapse_create2fb = pd.DataFrame(date_booking_seconds.iloc[:, 0] - created_date_seconds.iloc[:, 0],
                                    columns=['elapse_create2fb'])
    elapse_active2fb = pd.DataFrame(date_booking_seconds.iloc[:, 0] - first_active_seconds.iloc[:, 0],
                                    columns=['elapse_active2fb'])

    # One-hot-encoding features
    ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
                 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    df_all = pd.DataFrame()
    for f in ohe_feats:
        df_all_dummy = pd.get_dummies(data[f], prefix=f)
        df_all = pd.concat((df_all, df_all_dummy), axis=1)
    av = data.age.values
    data['age'] = np.where(np.logical_or(av < 10, av > 90), np.nan, av)
    df_all.set_index(data.index, inplace=True)
    re = pd.concat(
        [df_all, created_date_seconds, first_active_seconds, data['age']], axis=1)
    print re.columns
    return re


def get_one_hot_size_and_elapse(grouped, the_name):
    id_what_size = grouped.size().unstack()
    id_what_size.rename(columns=lambda x: the_name + '_' + x, inplace=True)
    id_what_size.fillna(0.0, inplace=True)

    id_what_elapse = grouped.sum().unstack()
    id_what_elapse.columns = id_what_elapse.columns.droplevel()
    id_what_elapse.columns.name = None
    id_what_elapse.rename(columns=lambda x: the_name + '_elp_' + x, inplace=True)
    return pd.concat([id_what_size, id_what_elapse], axis=1)


def get_session_vector(sessionCsv):
    print sessionCsv.columns
    id_action_grouped = sessionCsv.groupby(['user_id', 'action'])
    id_action_feats = get_one_hot_size_and_elapse(id_action_grouped, 'action')
    id_atype_grouped = sessionCsv.groupby(['user_id', 'action_type'])
    id_atype_feats = get_one_hot_size_and_elapse(id_atype_grouped, 'atype')
    id_adetail_grouped = sessionCsv.groupby(['user_id', 'action_detail'])
    id_adetail_feats = get_one_hot_size_and_elapse(id_adetail_grouped, 'adetail')
    id_device_grouped = sessionCsv.groupby(['user_id', 'device_type'])
    id_device_feats = get_one_hot_size_and_elapse(id_device_grouped, 'adevice')
    print id_action_feats.columns
    print id_action_feats.shape, id_atype_feats.shape, id_adetail_feats.shape, id_device_feats.shape
    id_groups = sessionCsv.groupby('user_id')
    id_size = pd.DataFrame(id_groups.size(), columns=["action_size"])
    id_elapse_sum = id_groups.sum()
    # id_elapse_sum['secs_elapsed'] = id_elapse_sum['secs_elapsed'].apply(lambda y: math.log(y) if pd.notnull(y) and y > 0 else y)
    id_elapse_sum.columns = ['action_elapsed_sum']
    session_vector = pd.concat(
        [id_elapse_sum, id_size, id_action_feats, id_atype_feats, id_adetail_feats, id_device_feats],
        axis=1)
    print session_vector.columns
    # session_vector.fillna(-1)
    # for index, row in session_vector.iterrows():
    #     print row, pd.notnull(row['secs_elapsed']), math.log(row['secs_elapsed'])

    return session_vector


trainCsv = pd.read_csv(TRAIN_CSV, header=0, index_col='id')
testCsv = pd.read_csv(TEST_CSV, header=0, index_col='id')
sessionCsv = pd.read_csv(SESSION_CSV, header=0)
print "sessonCsv.shape=", sessionCsv.shape
print "sessonCsv.columns=", sessionCsv.columns

startTime = dt.now()
destination = pd.DataFrame(trainCsv['country_destination'])
trainCsv.drop('country_destination', axis=1, inplace=True)
# idSave = testCsv['id']
pivTrain = trainCsv.shape[0]
# concat train and test, if make vector alternative, the vector will broken, so we must concat them.
dfAll = pd.concat((trainCsv, testCsv), axis=0)

dfAllVector = get_user_info_vector(dfAll)
sessionVector = get_session_vector(sessionCsv)
# print dfAllVector
dfAllVector = pd.concat((dfAllVector, sessionVector), axis=1, join_axes=[dfAllVector.index])
# print dfAllVector

x = dfAllVector[:pivTrain]
x_test = dfAllVector[pivTrain:]
# y2 = pd.DataFrame(destination, columns=['destination'])
y2 = destination
print "out to files!"
x.to_csv(TRAIN_FINAL_CSV)
x_test.to_csv(TEST_FINAL_CSV)
y2.to_csv(LABEL_FINAL_CSV)
print "over!"
