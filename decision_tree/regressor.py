from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

boston = load_boston()

regressor = DecisionTreeRegressor(random_state=0)
res = cross_val_score(regressor, boston.data, boston.target, cv=10)
print(res)