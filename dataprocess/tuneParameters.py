import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
import datetime as dt
from sklearn.model_selection import GridSearchCV

## Version 3 - LB 0.0644042
# Train month averages for test predictions seem work better than their linear fit,
# so I changed it (overfitting test data as hell... but who doesn't here? ;))

## Version 2 - LB 0.0644120
# LGBM performs much better, so I left him alone

## Version 1 - LB 0.0644711
# Both models have the same weight, which is based on cross-validation results, but
# XGB model seems to be worse on public LB, 'cause alone gets score 0.0646474,
# which is much worse than score of the combination. I reached the limit of submissions,
# so I will check how LGBM alone performs tomorrow. Check it out for your own ;)


print('Loading data...')
# properties = pd.read_csv('E:\\kaggle\\zillow_new\\properties_2016.csv', low_memory=False)
# train = pd.read_csv('E:\\kaggle\\zillow_new\\train_2016_v2.csv')
# sample_submission = pd.read_csv('E:\\kaggle\\zillow_new\\sample_submission.csv', low_memory=False)
# train = pd.merge(train, properties, how='left', on='parcelid')
# test = pd.merge(sample_submission[['ParcelId']], properties.rename(columns={'parcelid': 'ParcelId'}),
#                 how='left', on='ParcelId')
train = pd.read_csv('..\\data\\train_lgb.csv', low_memory=False)
# test = pd.read_csv("data\\test_lgb.csv", low_memory=False)
# del properties
# gc.collect();

print('Memory usage reduction...')
train[['latitude', 'longitude']] /= 1e6
# test[['latitude', 'longitude']] /= 1e6

train['censustractandblock'] /= 1e12
# test['censustractandblock'] /= 1e12

# for column in test.columns:
#     if test[column].dtype == int:
#         test[column] = test[column].astype(np.int32)
#     if test[column].dtype == float:
#         test[column] = test[column].astype(np.float32)

print('Feature engineering...')
train['month'] = pd.to_datetime(train['transactiondate']).dt.month
train = train.drop('transactiondate', axis=1)
from sklearn.preprocessing import LabelEncoder

non_number_columns = train.dtypes[train.dtypes == object].index.values

# for column in non_number_columns:
#     train_test = pd.concat([train[column], test[column]], axis=0)
#     encoder = LabelEncoder().fit(train_test.astype(str))
#     train[column] = encoder.transform(train[column].astype(str)).astype(np.int32)
#     test[column] = encoder.transform(test[column].astype(str)).astype(np.int32)

for column in non_number_columns:
    # train[column] = train[column].fillna(train[column].dropna().mode()[0])
    encoder = LabelEncoder().fit(train[column].astype(str))
    train[column] = encoder.transform(train[column].astype(str)).astype(np.int32)


feature_names = train.columns[2:]
feature_names = [feature for feature in feature_names if feature != 'month']

month_avgs = train.groupby('month').agg(['mean'])['logerror', 'mean'].values - train['logerror'].mean()

from sklearn.linear_model import LinearRegression

month_model = LinearRegression().fit(np.arange(4, 13, 1).reshape(-1, 1),
                                     month_avgs[3:].reshape(-1, 1))

print('Preparing arrays and throwing out outliers...')
X_train = train[feature_names].values
y_train = train.iloc[:, 1].values
# X_test = test[feature_names].values

# del test
gc.collect();

month_values = train['month'].values
X_train = np.hstack([X_train, month_model.predict(month_values.reshape(-1, 1))])

print((y_train > -0.4) & (y_train < 0.418))
X_train = X_train[ (y_train > -0.4) & (y_train < 0.4), :]
y_train = y_train[(y_train > -0.4) & (y_train < 0.4)]

print('Training LGBM model...')
ltrain = lgb.Dataset(X_train, label=y_train)

lgb_params = {
    # 'num_boost_round': 3000,
    'n_estimators': 500,
    # 'metric': 'mae',
    'max_depth': 20, # [5, 7, 9, 11]  11 ......0.05230 range(20, 28, 2) 20 0.0522987207084
    'num_leaves': 31, # [30, 50, 70] 30
    'min_child_weight': 5, # [5, 7, 9, 11]  11, range(11, 20, 1) 13 -0.0522928279209
    'min_child_samples': 13,
    'learning_rate': 0.01,
    'subsample': 0.85,
    'reg_lambda': 0.6, # -0.0522817311929
    'reg_alpha':0.8,
    'colsample_bytree': 0.95,
    # 'verbose': 0,
    'nthread': 4,
    'seed': 2017,
}


param_tests = {
    'reg_alpha': [i * 0.1 for i in range(0, 10, 2)],
    # 'num_leaves': [30, 50, 70]

}
# lgb_model = lgb.train(params, ltrain, verbose_eval=0, num_boost_round=2930)

# lgb_model = lgb.cv(lgb_params, ltrain, verbose_eval=10, num_boost_round=3000, early_stopping_rounds=100, nfold=5)
# print('Making predictions and praying for good results...')
model = GridSearchCV(estimator=lgb.LGBMRegressor(**lgb_params), param_grid=param_tests, scoring='neg_mean_absolute_error',  cv=5, verbose=10)
model.fit(X_train, y_train)
print(model.grid_scores_)
print(model.best_params_)
print( model.best_score_)