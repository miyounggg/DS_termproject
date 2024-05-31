import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data and fill missing values
health_stroke_df = pd.read_csv('C:\\Users\\choim\\DS_py\\15\\healthcare-dataset-stroke-data.csv')
health_stroke_df['bmi'].fillna(health_stroke_df['bmi'].median(), inplace=True)
health_stroke_df = health_stroke_df[health_stroke_df['smoking_status'] != 'Unknown']

# Define function to compute outlier thresholds
def outlier_thresholds(column):
    Q1 = health_stroke_df[column].quantile(0.25)
    Q3 = health_stroke_df[column].quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Cap outliers
for col in ['age', 'avg_glucose_level', 'bmi']:
    lb, ub = outlier_thresholds(col)
    health_stroke_df[col] = health_stroke_df[col].clip(lower=lb, upper=ub)

# Bin numeric variables
bin_dict = {
    'age': [0,10,20,30,40,50,60,70,80,90],
    'bmi': [10,20,30,40,50,60,100],
    'avg_glucose_level': [50,90,130,170,210,250,300]
}

for var, bins in bin_dict.items():
    health_stroke_df[f'{var}_group'] = pd.cut(health_stroke_df[var], bins)


# Prepare data for machine learning
# Categorical Variables One-Hot Encoding
nom_cols = ['gender', 'ever_married', 'work_type', 'Residence_type']
health_stroke_df = pd.get_dummies(health_stroke_df, columns=nom_cols)
# Converting smoking status to code = 'no info' 라는 라벨 생성
health_stroke_df['smoking_status_code'] = health_stroke_df['smoking_status'].map({
   'never smoked': 0, 'formerly smoked': 1, 'smokes': 2
})

# MinMax scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
health_stroke_df[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(health_stroke_df[['age', 'avg_glucose_level', 'bmi']])


health_stroke_df.to_csv("pre_drop.csv", index = False)







