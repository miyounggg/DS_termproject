# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import itertools

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis
from sklearn import metrics
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Ignore warnings in IPython
import warnings
warnings.filterwarnings('ignore')


# load the csv file and preview the basic information
health_stroke_df = pd.read_csv('C:\\Users\\choim\\DS_py\\15\\healthcare-dataset-stroke-data.csv')
health_stroke_df.info()

# summarising how many entries are missing
health_stroke_df[health_stroke_df.iloc[:,:] == 'Unknown'] = np.NaN
total_entries = len(health_stroke_df)
bmi_number_of_missing = health_stroke_df.bmi.isnull().sum()
smoke_number_of_missing = health_stroke_df.smoking_status.isnull().sum()

bmi_missing_percent = 100*bmi_number_of_missing/total_entries
smoke_missing_percent = 100*smoke_number_of_missing/total_entries

# print the missing value info
print(f"Missing entries details:\n")
print(f"bmi has {bmi_number_of_missing} ({bmi_missing_percent}%) missing values.")
print(f"smoking_status has {smoke_number_of_missing} ({smoke_missing_percent}%) missing values")

print("\n")
# Inspect the likely categorical variables
likely_cat_col = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in likely_cat_col:
    print(f"{col}:", health_stroke_df[col].unique())


# Inspect the likely boolean variables 불리안 변수인지 확인
likely_bool_col = ['hypertension', 'heart_disease', 'stroke']

for col in likely_bool_col:
    print(f"{col}:", health_stroke_df[col].unique())

# Summary stats for numerical columns, i.e [age, avg_glucose_level, bmi]
numeric_col = ['age', 'avg_glucose_level', 'bmi']
health_stroke_df[numeric_col].describe()

    
#음수인지확인
print("\n")
print(f"Number of negative value entries")
for num_var in numeric_col:
    non_neg_check = health_stroke_df[num_var].dropna().apply(lambda x: 0 if x>=0 else 1)
    print(f"{num_var}: {non_neg_check.sum()}")
    


health_stroke_df[health_stroke_df.work_type=='children'].age.max()

# Figures inline and set visualisation style 
# children의 나이를 확인하면서 진짜 어린이인지 확인
health_stroke_df[health_stroke_df.work_type=='children'].age.plot(kind='hist')
plt.xlabel('Age')
plt.title("Age distribution for work_type=children")
plt.show()
    
