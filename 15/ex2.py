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

# Handling missing values
health_stroke_df[health_stroke_df.iloc[:, :] == 'Unknown'] = np.NaN
health_stroke_df['bmi'].fillna(health_stroke_df['bmi'].median(), inplace=True)
health_stroke_df['smoking_status'].fillna('no info', inplace=True)


#--------------------------------------------------------------
# Identify outliers in 'age', 'avg_glucose_level', 'bmi'
# Using IQR method to define thresholds for outliers
def outlier_thresholds(column):
    Q1 = health_stroke_df[column].quantile(0.25)
    Q3 = health_stroke_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

# Calculate outlier thresholds for 'age', 'avg_glucose_level', 'bmi'
age_lb, age_ub = outlier_thresholds('age')
glucose_lb, glucose_ub = outlier_thresholds('avg_glucose_level')
bmi_lb, bmi_ub = outlier_thresholds('bmi')

# 이상치를 상위 및 하위 경계값으로 제한하기
# Cap the outliers instead of removing to avoid losing too much data
health_stroke_df['age'] = health_stroke_df['age'].clip(lower=age_lb, upper=age_ub)
health_stroke_df['avg_glucose_level'] = health_stroke_df['avg_glucose_level'].clip(lower=glucose_lb, upper=glucose_ub)
health_stroke_df['bmi'] = health_stroke_df['bmi'].clip(lower=bmi_lb, upper=bmi_ub)

# Check the dataset after filling and capping
health_stroke_df.describe()

#--------------------------------------------------------------

# EDA on categorical variables [hypertension, heart_disease, ever_married, 
# work_type, Residence_type, smoking_status]

def get_prob_per_class_within_one_cat_feat(feature_col, df):
    """
    Function that returns the probabilities of entries of a certain class in one 
    particular feature variable having the target=1 in the dataset df (i.e. a patient 
    having strokes before in this case)

    Parameters
    ----------
    feature_col: str
        The particular feature variable of interest.
    
    df : Dataframe
        The input dataframe containing the dataset.

    Returns
    -------
    prob_per_cat_class_df: Dataframe
        A dataframe with the probabilities of entries of a certain class in one 
    particular feature variable having the target=1


    """
    prob_per_cat_class_df = pd.DataFrame(columns=[feature_col, 'sample_size', 'prob of target=1'])
    class_label_list = []
    prob_list = []
    sample_size_list = []
    for class_label in df[feature_col].dropna().unique():
        stroke_tot = df[df[feature_col]==class_label].stroke.sum()
        tot_count = df[df[feature_col]==class_label].stroke.count()
        sample_size_list.append(tot_count)
        class_label_list.append(class_label)
        prob_list.append(stroke_tot/tot_count)
    prob_per_cat_class_df[feature_col] = class_label_list
    prob_per_cat_class_df['prob of target=1'] = prob_list
    prob_per_cat_class_df['sample_size'] = sample_size_list
    return prob_per_cat_class_df

cat_col = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
prob_df = {}
for cat_var in cat_col:
    print(f"Categorical variable: {cat_var}")
    prob_df[cat_var] = get_prob_per_class_within_one_cat_feat(cat_var, health_stroke_df)
    print(prob_df[cat_var])
    print("-"*30, "\n")

# defining a function that returns all the related probability bar plots of patient of a certain class in each 
# particular catorgical feature variable having strokes  

def plot_prob_per_cat_class(df, health_stroke_df, fig_hsize=30, fig_wsize=30, ncols=3, nrows=3, fontsize=20):
    """
    Function that returns bar plots of the probabilities of entries of a certain class in one 
    particular feature variable having the target=1 in the dataset df (i.e. a patient 
    having strokes before in this case)

    Parameters
    ----------    
    df : Dataframe
        The input dataframe containing the probability of target=1 per category class 
        information.

    Returns
    -------
    probability plots per categorical classes

    """
    # initialising the figure plots
    i = 1
    fig = plt.figure(figsize =(fig_hsize, fig_wsize))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    for cat_var in df.keys():
        ax = fig.add_subplot(nrows, ncols, i)
        df[cat_var][['prob of target=1']].plot(kind='bar', ax=ax, fontsize=fontsize)
        plt.hlines(health_stroke_df.stroke.mean(),-1, len(df[cat_var])+.5)
        plt.annotate('prob of target=1 \n for whole dataset', xy=(0.25, 0.018), xytext=(0.05, 0.01), 
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=fontsize)
        ax.set_ylabel('Probability',fontsize=fontsize)
        ax.set_xlabel(cat_var,fontsize=fontsize)
        plt.xticks(np.arange(len(df[cat_var][cat_var])), df[cat_var][cat_var])
        i+=1
    return fig

def plot_num_var_KDE(df, num_var):
    """
    Function that returns kernel density estimate (KDE) plot of the given numeric variable
    for each binary target groups in the dataset df (i.e. a patient 
    having strokes before in this case)

    Parameters
    ----------    
    df : Dataframe
        The input dataframe containing the (healthcare) dataset
        
    num_var : str
        The numeric variable to be plotted

    Returns
    -------
    kernel density estimate (KDE) plot of num_var per target class

    """
    facet_plot = sns.FacetGrid(df, hue='stroke', aspect=4)
    facet_plot.map(sns.kdeplot, num_var, shade=True)
    facet_plot.set(xlim=(0, df[num_var].max()))
    facet_plot.add_legend()
    plt.title(f'KDE plot of {num_var} per strokes class labels')
    return facet_plot


def plot_num_var_vio_strip(df, num_var):
    """
    Function that returns violin and strip plots of the given numeric variable
    for each binary target groups in the dataset df (i.e. a patient 
    having strokes before in this case)

    Parameters
    ----------    
    df : Dataframe
        The input dataframe containing the (healthcare) dataset
        
    num_var : str
        The numeric variable to be plotted

    """
    fig, ax =  plt.subplots(ncols=2, sharey=True, figsize =(12,5))
    sns.violinplot(x='stroke', y=num_var, data=df, ax=ax[0])
    ax[0].set_title(f'Distribution of {num_var} per strokes class labels\n (violin plot)')
    sns.stripplot(x='stroke', y=num_var, data=df, alpha=0.2, jitter=True, ax=ax[1])
    ax[1].legend(('no stroke','stroke'))
    ax[1].set_title(f'Distribution of {num_var} per strokes class labels\n (strip plot)')




# def bootstrap_replicate_1d(data, func):
#     return func(np.random.choice(data, size=len(data)))

# def draw_bs_reps(data, func, size=1):
#     """ 
#     Function that draw bootstrap replicates.

#     Parameters
#     ----------    
#     data : arr-like 
#         The input data to be investigated and bootstrapped.
        
#     func : func
#         Function on the bootstrap samples to return

#     Returns
#     -------
#     bs_replicates : 
#         bootstrap replicates

#     """

#     # Initialize array of replicates: bs_replicates
#     bs_replicates = np.empty(size)

#     # Generate replicates
#     for i in range(size):
#         bs_replicates[i] = bootstrap_replicate_1d(data, func)

#     return bs_replicates



# Bin the numeric variables into groups
age_bins = [0,10,20,30,40,50,60,70,80,90]
health_stroke_df['age_group'] = pd.cut(health_stroke_df.age, age_bins)
bmi_bins = [10,20,30,40,50,60,100]
health_stroke_df['bmi_group'] = pd.cut(health_stroke_df.bmi, bmi_bins)
avg_glucose_bins = [50,90,130,170,210,250,300]
health_stroke_df['avg_glucose_group'] = pd.cut(health_stroke_df.avg_glucose_level, avg_glucose_bins)

# inspecting the results
health_stroke_df.iloc[:,-3:].head()

# Calculating the probability of having stroke per each bin within one particular numeric column
num_bin_col = ['age_group', 'bmi_group', 'avg_glucose_group']
prob_num_df = {}
for num_bin_var in num_bin_col:
    print(f"Numerical variable: {num_bin_var.split('_group')[0]}")
    prob_num_df[num_bin_var] = get_prob_per_class_within_one_cat_feat(num_bin_var, health_stroke_df)
    print(prob_num_df[num_bin_var])
    print("-"*30, "\n")


health_stroke_cat_numeric_df = health_stroke_df.iloc[:,:-3].copy()
health_stroke_cat_numeric_df.drop(columns=['id'], inplace=True)

# convert the categorical variables to categorical type
# (for efficient reason)
cat_col = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
health_stroke_cat_numeric_df[cat_col] = health_stroke_cat_numeric_df[cat_col].astype('category')
health_stroke_cat_numeric_df.info()


# Creating a copy of the health stroke dataframe and imputing the missing values
# in ['bmi', 'smoking_status'] for calculating the correlation coefficient
# (Imputation methods will be explained in more detail in the next section)
health_stroke_df2 = health_stroke_df.iloc[:,:-3].copy()

# using median for impute bmi missing
health_stroke_df2.bmi.fillna(health_stroke_df2.bmi.median(), inplace=True)
# using a new label 'no_info' for impute smoking_status missing
health_stroke_df2.smoking_status.fillna('no info', inplace=True)


# Label encoding

nom_col = ['gender', 'ever_married', 'work_type', 'Residence_type']
ord_col = ['smoking_status']

health_stroke_df2 = pd.get_dummies(health_stroke_df2, columns = nom_col)

ord_var_code = 'smoking_status_code'
smoke_mapper = {'never smoked': 0, 'no info': 1,
                'formerly smoked': 2,'smokes':3}
health_stroke_df2[ord_var_code] = health_stroke_df2['smoking_status'].map(smoke_mapper)



health_stroke_df_temp = health_stroke_df.copy()
health_stroke_df_temp.smoking_status.fillna('NAN', inplace=True)

#for work in health_stroke_df_temp.work_type.unique(): 
#   sns.countplot(x='smoking_status', data=health_stroke_df_temp.groupby('work_type').get_group(work), 
#                 order=['never smoked', 'formerly smoked', 'smokes', 'NAN'])
#   plt.title(work)
#   plt.show()


#for work in health_stroke_df_temp.gender.unique(): 
#   sns.countplot(x='smoking_status', data=health_stroke_df_temp.groupby('gender').get_group(work),
#                 order=['never smoked', 'formerly smoked', 'smokes', 'NAN'])
#   plt.title(work)
#   plt.show()


# Preview again the encoded df
health_stroke_df2.head()
health_stroke_df2.info()

#dropping the id and information duplicating columns
health_stroke_df_ML = health_stroke_df2.drop(columns=['id', 'gender_Other', 'ever_married_No', 'Residence_type_Rural',
                                                      'work_type_Never_worked', 'smoking_status'])
health_stroke_df_ML.info()


numeric_col = ['age', 'avg_glucose_level', 'bmi']
health_stroke_df_ML_non_stand = health_stroke_df_ML.copy()
health_stroke_df_ML[numeric_col].describe()

#standardise / normalise the numeric variable such that they roughly lie within the range
min_max_scaler = MinMaxScaler()
health_stroke_df_ML[numeric_col] = min_max_scaler.fit_transform(health_stroke_df_ML[numeric_col])

print("\n")
health_stroke_df_ML[numeric_col].describe()