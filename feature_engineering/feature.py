import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

data = pd.read_csv('data/train.csv')
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

selector = VarianceThreshold()
X_var0 = selector.fit_transform(X)

X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
