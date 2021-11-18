import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
import sklearn

boston = load_boston()
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
score = cross_val_score(regressor, boston.data, boston.target, cv=10, scoring="neg_mean_squared_error").mean()

print(score)

# sklearn当中的模型评估指标
sorted(sklearn.metrics.SCORERS.keys())

