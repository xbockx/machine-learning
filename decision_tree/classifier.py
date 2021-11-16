from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz
import matplotlib.pyplot as plt

wine = load_wine()

pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3, random_state=0)

clf = tree.DecisionTreeClassifier(criterion="entropy"
                                  , random_state=30
                                  # , splitter="random"
                                  )
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)

print(score)
#
# feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']
# dot_data = tree.export_graphviz(clf
#                                 , feature_names=feature_name
#                                 , class_names=["琴酒", "雪莉", "贝尔摩德"]
#                                 , filled=True
#                                 , rounded=True
#                                 )
# graph = graphviz.Source(dot_data)
# # graph.view()
#
# print([*zip(feature_name, clf.feature_importances_)])

test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i+1,
                                      criterion="entropy",
                                      random_state=30)
    clf = clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)
    test.append(score)
plt.plot(range(1, 11), test, color="red", label="max_depth")
plt.legend()    #加图例
plt.show()