from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

pd.DataFrame(data)

# 归一化
scaler = MinMaxScaler()
scaler.fit(data)
result = scaler.transform(data)

result_ = scaler.fit_transform(data)
scaler.inverse_transform(result)

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler(feature_range=[5, 10])
result = scaler.fit_transform(data)

# 标准化
scaler = StandardScaler()
scaler.fit(data)
scaler.mean_
scaler.var_
x_std = scaler.transform(data)
x_std.mean()
x_std.std()
scaler.fit_transform(data)
scaler.inverse_transform(x_std)