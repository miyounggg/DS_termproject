import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data and fill missing values
health_stroke_df = pd.read_csv('C:\\Users\\choim\\DS_py\\16\\data.csv')
health_stroke_df['bmi'].fillna(health_stroke_df['bmi'].median(), inplace=True)
health_stroke_df['smoking_status'].fillna('no info', inplace=True)

# Outlier threshold function
def outlier_thresholds(column):
    Q1 = health_stroke_df[column].quantile(0.25)
    Q3 = health_stroke_df[column].quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Ceiling on age data
health_stroke_df['age'] = np.ceil(health_stroke_df['age'])

# Cap outliers
for col in ['age', 'bmi']:
    lb, ub = outlier_thresholds(col)
    health_stroke_df[col] = health_stroke_df[col].clip(lower=lb, upper=ub)

# One-hot encoding for categorical variables
nom_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
health_stroke_df = pd.get_dummies(health_stroke_df, columns=nom_cols)

# MinMax scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
health_stroke_df[['avg_glucose_level', 'bmi']] = scaler.fit_transform(health_stroke_df[['avg_glucose_level', 'bmi']])

# Calculate the correlation matrix
correlation_matrix = health_stroke_df.corr()

# Generate a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
