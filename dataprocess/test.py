import pandas as pd
import numpy as np

a = pd.DataFrame(data=[[1,2,3.0], [2.0, 4, 5], [4,5,6]], columns=['a', 'b', 'c'])
print(a.iloc[1:2, :])

# sample_submission = pd.read_csv('E:\\kaggle\\zillow_new\\sample_submission.csv', low_memory=False)
