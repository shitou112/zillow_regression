import pandas as pd

print('Loading data...')
properties = pd.read_csv('E:\\kaggle\\zillow_new\\properties_2016.csv', low_memory=False)

y_train = pd.read_csv('E:\\kaggle\\zillow_new\\train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
test = pd.read_csv('E:\\kaggle\\zillow_new\\sample_submission.csv', low_memory=False)
x_train = y_train.merge(properties, how='left', on='parcelid')


test['parcelid'] = test['ParcelId']
test.drop(['ParcelId'], axis=1, inplace=True)
x_test = test.merge(properties, how='left', on='parcelid')
x_test.drop(['201610', '201611', '201612', '201710', '201711', '201712'], axis=1, inplace=True)
print(x_train.shape)
print(x_test.shape)

# write data ...
x_train.to_csv('..\\data\\train_df.csv', index=False)

test_data_num = x_test.shape[0]
n_fold = 3
n = test_data_num // n_fold
print(n)
for j in range(n_fold):
    if j < n_fold-1:
        x_test.iloc[j*n: (j+1)*n, :].to_csv('..\\data\\test_df'+str(j)+'.csv', index=False)
    else:
        x_test.iloc[j*n: , :].to_csv('..\\data\\test_df'+str(j)+'.csv', index=False)