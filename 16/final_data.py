import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# CSV 파일을 읽어와 DataFrame으로 저장합니다.
data = pd.read_csv('C:\\Users\\choim\\DS_py\\16\\filled_bmi.csv')

# Define function to compute outlier thresholds
def outlier_thresholds(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


# Cap outliers
for col in ['age', 'bmi']:
    lb, ub = outlier_thresholds(col)
    data[col] = data[col].clip(lower=lb, upper=ub)


# MinMax scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['age', 'avg_glucose_level']] = scaler.fit_transform(data[['age', 'avg_glucose_level']])


data.to_csv("final_data.csv", index = False)

