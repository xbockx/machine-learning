from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = load_breast_cancer()

rfc = RandomForestClassifier(n_estimators=100, random_state=30)
score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean();

scorel = []
for i in range(165, 175):
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=30)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    scorel.append(score)

print(max(scorel), ([*range(165, 175)][scorel.index(max(scorel))]))
plt.figure(figsize=[20, 5])
plt.plot(range(165, 175), scorel)
plt.show()

# n_estimators=168

param_grid = {'min_samples_split': np.arange(2, 2+20, 1)}
rfc = RandomForestClassifier(n_estimators=168,
                             max_depth=6,
                             random_state=30)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)
