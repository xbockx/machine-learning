import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

dataset = load_boston()
x_full, y_full = dataset.data, dataset.target
n_samples = x_full.shape[0]
n_features = x_full.shape[1]

# 随机数种子
rng = np.random.RandomState(0)
missing_rate = 0.5

# np.floor向下取整，返回.0格式的浮点数
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))

# randint(下限， 上线， n) 在下限和上限之间取出n个整数
missing_features = rng.randint(0, n_features, n_missing_samples)
missing_samples = rng.randint(0, n_samples, n_missing_samples)

x_missing = x_full.copy()
y_missing = y_full.copy()

x_missing[missing_samples, missing_features] = np.nan
x_missing = pd.DataFrame(x_missing)

# 使用均值进行填补
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x_missing_mean = imp_mean.fit_transform(x_missing)

# 使用0进行填补
imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
x_missing_0 = imp_0.fit_transform(x_missing)


x_missing_reg = x_missing.copy()
sortindex = np.argsort(x_missing_reg.isnull().sum(axis=0)).values

for i in sortindex:
    df = x_missing_reg
    fillc = df.iloc[:, i]
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)

    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

    Ytrain = fillc[fillc.notnull()]
    Ytest = fillc[fillc.isnull()]
    Xtrain = df_0[Ytrain.index, :]
    Xtest = df_0[Ytest.index, :]

    rfc = RandomForestRegressor(n_estimators=100)
    rfc = rfc.fit(Xtrain, Ytrain)
    Ypredict = rfc.predict(Xtest)

    x_missing_reg.loc[x_missing_reg.iloc[:, i].isnull(), i] = Ypredict

print(x_missing_reg)

X = [x_full, x_missing_mean, x_missing_0, x_missing_reg]

mse = []

for x in X:
    estimator = RandomForestRegressor(n_estimators=100, random_state=0)
    scores = cross_val_score(estimator, x, y_full, scoring='neg_mean_squared_error', cv=5).mean()
    mse.append(scores * -1)

print(mse)

x_labels = ['Full data', 'Zero Imputation', 'Mean Imputation', 'Regressor Imputation']
colors = ['r', 'g', 'b', 'orange']

plt.figure(figsize=(12, 6))         # 画出画布
ax = plt.subplot(111)       #添加子图
for i in np.arange(len(mse)):
    ax.barh(i, mse[i], color=colors[i], alpha=0.6, align='center')
ax.set_title('Imputation Techniques with Boston Data')
ax.set_xlim(left=np.min(mse) * 0.9,
            right=np.max(mse) * 1.1)
ax.set_yticks(np.arange(len(mse)))
ax.set_xlabel('MSE')
ax.set_yticklabels(x_labels)
plt.show()

