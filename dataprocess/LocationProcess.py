import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

important_features = ['calculatedfinishedsquarefeet', 'taxamount', 'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
                      'landtaxvaluedollarcnt']

def cityProcess(x_train, attribute, groupAttr):
    print(x_train.shape)
    map_value = x_train[[attribute, groupAttr]]\
        .groupby([groupAttr]).mean()
    key = map_value.index.values
    value = np.squeeze(map_value.values)
    map_value = dict(zip(key, value))
    x_train[attribute+'_'+groupAttr] = x_train[groupAttr].map(map_value)


    # map_value = x_train[[attribute, groupAttr]] \
    #     .groupby([groupAttr]).mean().reset_index()
    # map_value.columns = [groupAttr, attribute + '_' + groupAttr]
    # x_train = pd.merge(x_train, map_value, on=[groupAttr], how='left')

    print(x_train.shape)
    return x_train

def yearGroup(x_train, attribute):
    map_value = x_train[[attribute, 'yearbuilt']]\
        .groupby(['yearbuilt']).mean().reset_index()
    map_value.columns = ['yearbuilt', attribute+'_year_mean']
    x_train = pd.merge(x_train, map_value, on=['yearbuilt'], how='left')
    return x_train

def locationProcess(x_train):
    # k_means cluster

    x_train['location'] = KMeans(n_clusters=1000, random_state=0).fit_predict(x_train[['latitude', 'longitude']])
    for col in important_features:
        map_value = x_train[['location', 'calculatedfinishedsquarefeet']].groupby(['location']).mean()
        map_value = map_value.reset_index()
        map_value.columns = ['location', col+'_mean']
        x_train = x_train.merge(map_value, on=['location'], how='left')

    print(x_train.head())

if __name__ == '__main__':
    x_train = pd.read_csv('data\\train_lgb.csv', low_memory=False, iterator=True)
    # x_test = pd.read_csv("data\\test_lgb.csv", low_memory=False)
    y_train = pd.read_csv('E:\\kaggle\\zillow_new\\train_2016_v2.csv', iterator=True, parse_dates=['transactiondate'], low_memory=False)
    print('--aaaaaa----')
    x_train = x_train.get_chunk(1000)
    y_train = y_train.get_chunk(1000)
    locationProcess(x_train)
    # x_train['calculatedfinishedsquarefeet_mean']
    # print(x_train['calculatedfinishedsquarefeet_mean'])

