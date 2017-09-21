import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import dataprocess.LocationProcess as location

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split



xgb_month_params = {
    'n_estimators': 100,
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_child_weight': 3,
    'gamma': 0,
    'subsample': 0.85,
    'colsample_bytree': 0.95,
    # 'reg_alpha': 0.005,
    'eval_metric': 'rmse',
    'nthread': 4,
    'seed': 2017,
    'silent':1
}
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 20,
    'min_child_weight': 5,
    'gamma': 0,
    'subsample': 0.85,
    'colsample_bytree': 0.95,
    'reg_lambda': 0.6,
    'eval_metric': 'mae',
    'nthread': 4,
    'seed': 2017,
    'silent':1
}

lgb_params = {
    # 'num_boost_round': 3000,
    'n_estimators': 500,
    'metric': 'mae',
    'max_depth': 20, # [5, 7, 9, 11]  11 ......0.05230 range(20, 28, 2) 20 0.0522987207084
    'num_leaves': 31, # [30, 50, 70] 30
    'min_child_weight': 5, # [5, 7, 9, 11]  11, range(11, 20, 1) 13 -0.0522928279209
    'min_child_samples': 13,
    'learning_rate': 0.01,
    'subsample': 0.85,
    'reg_lambda': 0.6, # -0.0522817311929
    'reg_alpha':0.8,
    'colsample_bytree': 0.95,
    'verbose': 0,
    'nthread': 4,
    'seed': 2017,
}



param_tests = {
    'min_child_weight': [2, 6, 10],
    'min_child_samples': [5, 10, 20],
}

result_index = 0
x_train = pd.read_csv('..\\data\\train_df.csv', low_memory=False)
x_test = pd.read_csv("..\\data\\test_df"+str(result_index)+".csv", low_memory=False)
y_train = pd.read_csv('E:\\kaggle\\zillow_new\\train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
result = pd.DataFrame(np.zeros((x_test.shape[0], 7)), columns=['parcelid', '201610', '201611', '201612', '201710', '201711', '201712'])
result['parcelid'] = x_test['parcelid'].values

print(result.tail())
print(x_train.shape)
print(x_test.shape)
# drop useless feature
x_train.drop(['parcelid'], axis=1, inplace=True)
x_test.drop(['parcelid'], axis=1, inplace=True)

def typeTrans(data: pd.DataFrame):
    for col in data.columns:
        if data[col].dtype == np.float64:
            data[col] = data[col].astype(np.float32)
        elif data[col].dtype == np.int64:
            data[col] = data[col].astype(np.int32)

        return data

x_train = typeTrans(x_train)
y_train = typeTrans(y_train)
x_test = typeTrans(x_test)

print('删除异常点')
x_train = x_train.loc[((y_train['logerror']) > -0.3) & ((y_train['logerror']) < 0.3), :]
y_train= y_train.loc[((y_train['logerror']) > -0.3) & ((y_train['logerror']) < 0.3), :]

index = x_train['calculatedfinishedsquarefeet']<13000
x_train = x_train.loc[index, :]
y_train = y_train.loc[index, :]


x_train = x_train.reset_index()
y_train = y_train.reset_index()
x_train.drop(['index'], axis=1, inplace=True)
y_train.drop(['index'], axis=1, inplace=True)

x_train.drop(['transactiondate', 'logerror'], axis=1, inplace=True)
train_nums = x_train.shape[0]



x_train['abs_logerror'] = np.abs(y_train['logerror'])

print(x_train.shape)
# print(x_test.shape)
x_train = x_train.append(x_test)

del x_test
gc.collect()


print('降低大数值')
x_train[['latitude', 'longitude']] /= 1e6
x_train['censustractandblock'] /= 1e12

# 填补空值
# (0.0521833544848, 0.0521497370622)
# knn_model = KNeighborsClassifier(n_neighbors=3)
# knn_model.fit(x_train.loc[~x_train['regionidzip'].isnull(), ['latitude', 'longitude']], x_train.loc[~x_train['regionidzip'].isnull() ,'regionidzip'])
# x_train.loc[x_train['regionidzip'].isnull(), 'regionidzip'] = knn_model.predict(x_train.loc[x_train['regionidzip'].isnull(), ['latitude', 'longitude']])



# 添加房屋使用年限属性 （不使用0.0521131， 使用0.0521146， 不推荐）
# x_train['user_years'] = 2016 - x_train['yearbuilt']

# (使用0.0523576747394， 不使用0.0523704981121， 使用效果更好)
x_train['taxdelinquencyflag'].fillna('N')
x_train = pd.concat([x_train, pd.get_dummies(x_train['fips'])], axis=1)

# (0.0521497370622, 0.0522153081385)
x_train = pd.concat([x_train, pd.get_dummies(x_train['propertylandusetypeid'])], axis=1)

# (使用0.0523452391788， 不使用0.0523576747394， 使用效果更好)
x_train['room_num'] = x_train['bedroomcnt'] + x_train['bathroomcnt']

# (0.0523369482222, 0.0523452391788)
x_train['living_all_rate'] = x_train['finishedsquarefeet12'] / x_train['lotsizesquarefeet']
x_train['N-LivingAreaProp2'] = x_train['finishedsquarefeet12']/x_train['finishedsquarefeet15']


#Amout of extra space(0.0520669555005, 0.0521634656378)
x_train['N-ExtraSpace'] = x_train['lotsizesquarefeet'] - x_train['calculatedfinishedsquarefeet']
x_train['N-ExtraSpace-2'] = x_train['finishedsquarefeet15'] - x_train['finishedsquarefeet12']

x_train['square_per'] = x_train['calculatedfinishedsquarefeet'] / x_train['room_num']

x_train["N-location"] = x_train["latitude"] + x_train["longitude"]
x_train["N-location-2"] = x_train["latitude"]*x_train["longitude"]
x_train["N-location-2round"] = x_train["N-location-2"].round(-4)

x_train["N-latitude-round"] = x_train["latitude"].round(-4)
x_train["N-longitude-round"] = x_train["longitude"].round(-4)


#Number of properties in the zip(0.0530555574765)
zip_count = x_train['regionidzip'].value_counts().to_dict()
x_train['N-zip_count'] = x_train['regionidzip'].map(zip_count)

#Number of properties in the city(使用0.0530163393381，)
city_count = x_train['regionidcity'].value_counts().to_dict()
x_train['N-city_count'] = x_train['regionidcity'].map(city_count)

#Number of properties in the city
region_count = x_train['regionidcounty'].value_counts().to_dict()
x_train['N-county_count'] = x_train['regionidcounty'].map(city_count)


# # 修正偏度属性（）
x_train['calculatedfinishedsquarefeet'] = np.log(x_train['calculatedfinishedsquarefeet'])
# x_train['finishedsquarefeet12'] = np.log(x_train['finishedsquarefeet12'])



# 根据位置信息补全(0.0522890539257, 0.0523369482222)
print('----location-----')
important_features = ['calculatedfinishedsquarefeet', 'taxamount',
                      'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
                      'landtaxvaluedollarcnt', 'yearbuilt',
                      'living_all_rate', 'room_num',
                      ]


for col in important_features:
    x_train = location.cityProcess(x_train, col, 'regionidzip')


for col in x_train.columns:
    if x_train[col].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(x_train[col].astype(str))
        x_train[col] = lbl.transform(x_train[col].astype(str)).astype(np.int32)

x_test = x_train[train_nums: ]
x_train = x_train[: train_nums]

# 验证集
# test_data = x_train.loc[y_train['transactiondate'] > '2016-10-15', :]
# test_y = y_train.loc[y_train['transactiondate'] > '2016-10-15', :]
#
#
# x_train = x_train.loc[y_train['transactiondate'] <= '2016-10-15', :]
# y_train = y_train.loc[y_train['transactiondate'] <= '2016-10-15', :]
#
# tmp_x, final_x, tmp_y, final_y = train_test_split(test_data, test_y, test_size=0.3, random_state=0)
# x_train = pd.concat([x_train, tmp_x], axis=0)
# y_train = pd.concat([y_train, tmp_y], axis=0)



x_train['transaction_month'] = y_train['transactiondate'].dt.month
abs_month_avgs = x_train.groupby('transaction_month').agg(['mean'])['abs_logerror', 'mean'].values - x_train['abs_logerror'].mean()
x_train.drop(['abs_logerror'], axis=1, inplace=True)
x_test.drop(['abs_logerror'], axis=1, inplace=True)



print(abs_month_avgs)
from sklearn.linear_model import LinearRegression


month_values = x_train['transaction_month'].values
month_model = LinearRegression().fit(np.arange(4, 13, 1).reshape(-1, 1), abs_month_avgs[3:].reshape(-1 ,1))
x_train['month_logerror_diff'] = month_model.predict(month_values.reshape(-1, 1))

x_train = x_train.fillna(-1)
x_train['logerror'] = y_train['logerror'].values
x_train.to_csv('..\\data\\x_train.csv', index=False)
x_train.drop(['logerror'], axis=1, inplace=True)


# final_x['transaction_month'] = final_y['transactiondate'].dt.month
# final_x['month_logerror_diff'] = month_model.predict(final_x['transaction_month'].values.reshape(-1,1))
#

# drop useless feature
# useless_feature = ['parcelid', 'buildingclasstypeid', 'finishedsquarefeet13', 'basementsqft', 'storytypeid', 'yardbuildingsqft26', 'fireplaceflag', 'architecturalstyletypeid', 'typeconstructiontypeid', 'finishedsquarefeet6', 'decktypeid', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'taxdelinquencyflag', 'taxdelinquencyyear', 'hashottuborspa', 'yardbuildingsqft17']
# x_train.drop(useless_feature, axis=1, inplace=True)


# mean_values = x_train.mean(axis=0)
# x_train = x_train.fillna(mean_values, inplace=True)




print('模型训练')
# y_train['logerror'] = y_train['logerror'] * 0.99

# model = GridSearchCV(estimator=xgb.XGBRegressor(**xgb_params), param_grid=param_tests, scoring='mean_absolute_error',  cv=5, verbose=10)
# model.fit(x_train, y_train['logerror'])
# print(model.grid_scores_, model.best_params_, model.best_score_)
# dtrain = xgb.DMatrix(data=x_train, label=y_train['logerror'], feature_names=x_train.columns)
# model = xgb.cv(xgb_params, dtrain, nfold=5, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=10)

# model = GridSearchCV(estimator=lgb.LGBMRegressor(**lgb_params), param_grid=param_tests, scoring='neg_mean_absolute_error',  cv=5, verbose=10)
# model.fit(x_train, y_train['logerror'])
# print(model.grid_scores_)
# print(model.best_params_)
# print( model.best_score_)


# model = lgb.cv(params, lgbtrain, nfold=5, num_boost_round=2000, verbose_eval=10)
# mults = []
# for i in range(10):
#     mults.append(1.000+0.0001*i)
# for mult in mults:
#     print(mult)
#     y_value = y_train['logerror'] * mult

# xgbtrain = xgb.DMatrix(data=x_train, label=y_train['logerror'])
# cv_model = xgb.cv(xgb_params, xgbtrain, num_boost_round=1000, verbose_eval=10)
# cv_model.nu
# model = xgb.train(xgb_params, xgbtrain, num_boost_round=1000, verbose_eval=10)

lgbtrain = lgb.Dataset(x_train, y_train['logerror'])
cv_model = lgb.cv(lgb_params, lgbtrain, num_boost_round=1000, verbose_eval=10, nfold=4)

model = lgb.train(lgb_params, lgbtrain, num_boost_round=820, verbose_eval=10)
# pred = model.predict(x_train)
# predict = model.predict(final_x)
# print(mean_absolute_error(predict, final_y['logerror']))
# lgb.plot_importance(model, max_num_features=30)
# plt.show()

# lgb.cv(lgb_params, lgbtrain, num_boost_round=2000, verbose_eval=10, nfold=4)
# lgb_model = lgb.train(params, lgbtrain, num_boost_round=2000, verbose_eval=10)
# lgb.plot_importance(lgb_model, max_num_features=30)
# plt.show()
# pred = lgb_model.predict(test_data)
# print(mean_absolute_error(pred, np.array(test_y['logerror'])))

# sample_submission = pd.read_csv('E:\\kaggle\\zillow_new\\sample_submission.csv', low_memory=False)
# sample_submission = typeTrans(sample_submission)
#



folds = 20
n = int(x_test.shape[0] / folds)

# del x_train
# gc.collect()
#
# for month in [10, 11, 12]:
#     print('iteration: month is '+ str(month) +' ....')
#     x_test['transaction_month'] = month
#     x_test['month_logerror_diff'] = month_model.predict(x_test['transaction_month'].values.reshape(-1, 1))
#
#     x_test.to_csv('..\\data\\x_test_' +str(result_index)+'_'+ str(month) + '.csv', index=False)
#     print(x_test.shape)
#
#     y_pred = model.predict(x_test)
#     print(y_pred.shape)
#     result['2016'+str(month)] = y_pred
#     result['2017'+str(month)] = y_pred
#
# #
# result.to_csv("result"+str(result_index)+".csv", index=False)
print('Saving predictions...')


# sample_submission.to_csv('lgb_submission.csv', index=False, float_format='%.6f')
print('Done!')