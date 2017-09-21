import pandas as pd
import numpy as np

a = pd.DataFrame(data=[[1,2,3.0], [2.0, 4, 5]], columns=['a', 'b', 'c'])
print(a.dtypes)
for i in a.columns:
    if a[i].dtype == np.float64:
        a[i] = a[i].astype(np.float16)

print(a[['a', 'b']].groupby('a').mean())
print(a['a'].value_counts())
