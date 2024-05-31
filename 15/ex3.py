import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data and fill missing values
health_stroke_df = pd.read_csv('C:\\Users\\choim\\DS_py\\15\\healthcare-dataset-stroke-data.csv')
health_stroke_df['bmi'].fillna(health_stroke_df['bmi'].median(), inplace=True)
health_stroke_df['smoking_status'].fillna('no info', inplace=True)
health_stroke_df.head()

#나이 데이터 중 소수점을 가진 이상치 데이터 올림
health_stroke_df['age'] = np.ceil(health_stroke_df['age'])

# Define function to compute outlier thresholds
def outlier_thresholds(column):
    Q1 = health_stroke_df[column].quantile(0.25)
    Q3 = health_stroke_df[column].quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Cap outliers
for col in ['age', 'bmi']:
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

# Function to calculate stroke probability per category
def get_prob_per_class_within_one_cat_feat(feature_col, df):
    results = []
    class_labels = df[feature_col].dropna().unique()
    for label in class_labels:
        sub_df = df[df[feature_col] == label]
        results.append({
            feature_col: label,
            'sample_size': sub_df.shape[0],
            'prob of target=1': sub_df['stroke'].mean()
        })
    return pd.DataFrame(results)


# Prepare data for machine learning
# Categorical Variables One-Hot Encoding
nom_cols = ['gender', 'ever_married', 'work_type', 'Residence_type','smoking_status']
health_stroke_df = pd.get_dummies(health_stroke_df, columns=nom_cols)


# MinMax scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
health_stroke_df[[ 'avg_glucose_level', 'bmi']] = scaler.fit_transform(health_stroke_df[[ 'avg_glucose_level', 'bmi']])


health_stroke_df.to_csv("pre_info.csv", index = False)
