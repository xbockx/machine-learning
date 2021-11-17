import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np


# import dataset
data = pd.read_csv('./data/train.csv')

# 筛选特征
data.drop(['Cabin', 'Name', 'Ticket'], inplace=True, axis=1)

# 处理缺失值
data["Age"] = data["Age"].fillna(data["Age"].mean())
data = data.dropna()

labels = data["Embarked"].unique().tolist()
data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))
data["Sex"] = (data["Sex"] == "male").astype("int")
# data.loc[:, "Sex"]


x = data.iloc[:, data.columns != "Survived"]
y = data.iloc[:, data.columns == "Survived"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3)

# 重排索引
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])

clf = DecisionTreeClassifier(criterion="entropy", random_state=30)
clf = clf.fit(Xtrain, Ytrain)
# score = clf.score(Xtest, Ytest)

# 交叉验证
score = cross_val_score(clf, x, y, cv=10).mean()
print(score)

tr = []
te = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25,
                                 max_depth=i+1,
                                 criterion="entropy")
    clf = clf.fit(Xtrain, Ytrain)
    score_tr = clf.score(Xtrain, Ytrain)
    score_te = cross_val_score(clf, x, y, cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))
plt.plot(range(1, 11), tr, color="red", label="train")
plt.plot(range(1, 11), te, color="blue", label="test")
plt.xticks(range(1,11))     # 刻度
plt.legend()
plt.show()

# 网格搜索：能够帮助我们同时调整多个参数的技术，枚举技术
gini_thresholds = np.linspace(0,0.5, 20)
# entropy_thresholds = np.linspace(0, 1, 50)

# 一串参数和这些参数对应的，我们希望网格搜索来搜索的参数的取值范围
params = {
    "criterion": ("gini", "entropy"),
    "splitter": ("best", "random"),
    "max_depth": [*range(1, 10)],
    "min_samples_leaf": [*range(1, 50, 5)],
    "min_impurity_decrease": [*np.linspace(0, 0.5, 20)]
}

clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, params, cv=10)
GS = GS.fit(Xtrain, Ytrain)

print(GS.best_params_)
print(GS.best_score_)
