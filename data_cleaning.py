# import packages and libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

#Using panda to read csv file into a dataframe
df = pd.read_csv('Data_(5).csv', index_col = 'id')
print(df.info())

#select loan_status column as our target variable and inspect the distribution of the values
print('\n')
print('Loan status distribution:')
print(df['loan_status'].value_counts())

'''Since we are only focus on the Fully Paid and Default loan for our prediction.
I will select only contains rows that have either Default or Fully Paid''' 

#new df with only fully paid and default loans
df = df.loc[df['loan_status'].isin(['Fully Paid','Default'])]
print('\n')
print('Dimension of the new dataframe: ')
print(df.shape)
print('Distribution of the total data:')
print(df['loan_status'].value_counts(normalize = True)) #66% of samples are fully paid and 34% of samples are default

#1. CLEANING DATA
#function to drop columns
def drop_cols(cols):
    df.drop(cols, axis =1, inplace = True)

'''Because the model is set to predict if a new borrowers will pay off or default their loan before making the decision to lend the loan.
By inspecting through the definition of each feature, I select to drop the following cols'''
#drop columns contain after-loan features
after_loan_cols = ['last_fico_range_low','last_fico_range_high', 'last_credit_pull_d']
drop_cols(after_loan_cols)

#inspecting addr_state column
x = df['addr_state'].loc[df['loan_status'] == 'Default'].value_counts()
y = df['addr_state'].loc[df['loan_status'] == 'Fully Paid'].value_counts()
print(x.corr(y)) #0.98 correlation
'''Because the amount of default loan seems to be propotional to fully paid loan per each state. 
This feature does not contribute to the overall prediction of the model'''
#drop addr_state
drop_cols('addr_state')

#function to determine columns with missing values
def miss_val(df):
    '''Determine number of percentage of missing values in a columnn'''
    #number of total missing value in the column
    miss_val = df.isnull().sum()
    #percent of total missing value
    miss_val_percent = 100*(df.isnull().sum())/len(df)
    #missing value type
    miss_val_type = df.dtypes
    #create a missing value table
    miss_val_table = pd.concat([miss_val, miss_val_percent, miss_val_type], axis =1).rename(columns= {0: 'total missing values', 1:'total % of missing value', 2: 'data type'})
    return miss_val_table
print('Missing Table')
print(miss_val(df))

#select columns that have more than 50% of missing values to drop from dataframe
dropped_features_list = sorted(df.isnull().sum()[df.isnull().mean() > 0.5].index) 
#drop columns that have too many missing values (>50%)
drop_cols(dropped_features_list)

'''there are around 1000 NaN values (less than 5% of the total instances) in emp_length column''' 
#inspect NaN distribution in 'emp_length'
print('\n')
print('Distribution of NaN value in emp_length')
print(df.groupby('loan_status')['emp_length'].apply(lambda x: x.isnull().value_counts()))

'''the distribution of the NaN value is not much different between default and fully paid loan'''
#drop all NaN value in 'emp_length'
df = df.dropna(subset=['emp_length'])
print('\n')
print('Current shape of the dataframe:')
print(df.shape)

#2. FEATURE ENGINEERING

#Convert emp_length data from object to numeric data based on description
#convert all 10+ years data to 10 years
df['emp_length'].replace('10+ years', '10 years', inplace = True)
#convert all '< 1 year' data to 0 year
df['emp_length'].replace('< 1 year', '0 year', inplace= True)
#convert all to numeric
df['emp_length'] = df['emp_length'].map(lambda x: float(str(x).split( )[0]))

#Convert term data from object to numeric data
df['term'] = df['term'].map(lambda y: float(str(y).split( )[0]))

#convert salary verification into numeric data. Verified/Source Verified (Yes):1 Not Verified (No): 0
df['salary_verified'] = df['verification_status'].apply(lambda x: np.float(np.logical_or(x == 'Verified', x=='Source Verified')))
drop_cols('verification_status')

''' Although, column 'mths_since_last_delinq' contains almost 47% missing value, this feature can be important to determine loan default. 
We will create new column (delinq_before) and convert all missing values of 'mths_since_last_delinq' to 0 as the borrower does not have any delinquent.
Other values in the column will be convert to 1 as the  borrower has deliquent
'''
#convert 'mths_since_last_delinq' column into 'delinq_before' with Yes: 1, No: 0
df['mths_since_last_delinq'].loc[df['mths_since_last_delinq'].notnull()] = 1
df['mths_since_last_delinq'].loc[df['mths_since_last_delinq'].isnull()] = 0
df['delinq_before'] = df['mths_since_last_delinq']
drop_cols('mths_since_last_delinq')

#3. CREATE NEW FEATURES BY COMBINING OLD FEATURES

#Inspect correlation of loan_amnt, installment, and term
loan_cols = ['loan_amnt','installment', 'term']
print('\n')
print(df[loan_cols].corr())
#Because loan_amnt, installment and term can be used to calculate interest_rate of a loan, create new interest_rate feature
df['interest_rate']=100*(df['term']*df['installment']-df['loan_amnt'])/(df['loan_amnt'])
#Drop term and loan_amnt to prevent noise
drop_cols(['loan_amnt', 'term'])

#Convert both issue_d and earliest_cr_line to datetime and extract the year
df['issue_year'] = pd.to_datetime(df['issue_d'], format = '%m/%d/%Y').dt.year
df['earliest_cr_line_yr'] = pd.to_datetime(df['earliest_cr_line'], format = '%m/%d/%Y').dt.year
#Since credit length of the borrower may have affect on their risk to default or payoff loan
#Calculate credit length from earliest_cr_line_year and issue_year
df['credit_length'] = df['issue_year'] - df['earliest_cr_line_yr']
drop_cols(['issue_year', 'issue_d','earliest_cr_line_yr', 'earliest_cr_line'])

#Because there are two FICO range score, I inspect correlation of fico_rang_low and fico_rang_high
fico_cols = ['fico_range_low','fico_range_high']
print('\n')
print(df[fico_cols].corr())
#These two value are highly correlated with each other. I will take a average of two and create a new col as fico_score
df['fico_score'] = (df['fico_range_low']+ df['fico_range_high'])/2
drop_cols(fico_cols)

#Inspect the shape of the dataframe
print('\n')
print('Current shape of the dataframe:')
print(df.shape)

#EXPLOLATORY DATA ANALYSIS
'''I will perform EDA on all numeric columns of the dataframe
Comparing the means of Default group and Fully-paid group using Student t-test
Determine if there is significant difference using p-value
Null Hypothesis: No difference between the means of two groups'''

#list of all numeric columns
numeric_cols = ['installment','interest_rate','emp_length','annual_inc','dti','fico_score','credit_length','acc_now_delinq','delinq_amnt','delinq_2yrs','inq_last_6mths']
#corvariance matrix of all numeric columns
print('\n')
print('COVARIANCE MATRIX OF ALL NUMERIC COLUMNS')
print(df[numeric_cols].cov())

#function to calculate t-test
def run_ttest(feature, col = 'loan_status', value1 = 'Fully Paid', value2 = 'Default'):
    '''Calculate t-statistics and p-value of t-test'''
    group1 = df.loc[df[col]== value1, feature]
    group2 = df.loc[df[col] == value2, feature]
    return ttest_ind(group1,group2)

#summary of all numeric features
for col in numeric_cols:
    print('\n')
    print('Summary of '+ col)
    print(run_ttest(col))
    print(df.groupby('loan_status')[col].describe(), '\n')

'''For binary columns that inlcude only binary value (0 or 1),
I chose to use Ztest to test null hypothesis and determine p value
Null Hypothesis: No difference between the proportions of two groups'''

#function to calculate z-test
def run_proportion_z_test(feature, col = 'loan_status', value1 = 'Fully Paid', value2 = 'Default'):
    '''Calculate z-statistics and p-value of z-test. Feature is the feature that z-test will be performed on'''
    group1 = df.loc[df[col]== value1, feature]
    group2 = df.loc[df[col] == value2, feature]
    n1 = len(group1)
    p1 = group1.sum()
    n2 = len(group2)
    p2 = group2.sum()
    z_score, p_value = proportions_ztest([p1, p2], [n1, n2])
    return('z-score = {}; p-value = {}'.format(z_score, p_value))

#summary of binary feature
binary_cols = ['salary_verified', 'delinq_before']
for col in binary_cols:
    print('\n')
    print('Summary of '+ col + ' (Yes:1, No: 0)')
    print(run_proportion_z_test(col))
    print(df.groupby('loan_status')[col].value_counts(normalize = True))  

'''For categorical columns,
I chose to use chi2test to test null hypothesis and determine p value
Null Hypothesis: No association between the two features'''

def run_chi2_test(feature, n, col = 'loan_status', value1 = 'Fully Paid', value2 = 'Default'):
    '''Calculate chi2 statistics and p-value. Feature is the feature that z-test will be performed on, n is number of categories'''
    group1 = df.loc[df[col] == value1,feature].value_counts().tolist()[:n] #select only the first 4 categories
    group2 = df.loc[df[col] == value2,feature].value_counts().tolist()[:n] #select only the first 4 categories
    chi2, p, dof, expected = chi2_contingency([group1, group2])
    print("chi-square test statistic:", chi2)
    print("p-value", p, '\n')

#inspect categorical column especially loan purpose and home_ownership
categorical_cols = ['purpose', 'home_ownership']
for category in categorical_cols:
    print('\n')
    print('Summary of '+ category)
    if category == 'purpose':
        print(run_chi2_test (category, 12))
    else:  
        print(run_chi2_test (category, 4))
    print(df.groupby('loan_status')[category].value_counts(normalize = True))
  
#FEARTURE ENGINEERING TO GET READY FOR MODEL LEARNING
#create dummies values for home_ownership and purpose features
df = pd.get_dummies(df, columns=['purpose', 'home_ownership'], drop_first = True)

#Convert loan_status column into numerical data. Fully_Paid : 0, Default: 1
df['default'] = df['loan_status'].apply(lambda x: np.float(x == 'Default'))
drop_cols('loan_status')

#inspect data frame that is ready for machine learning
print('\n')
print('FINAL DIMENSION OF DATAFRAME')
print(df.shape)
print('\n')
print('INSPECT FINAL DATAFRAME')
print(df.info())
print('\n')
print('Data distribution ')
print(df['default'].value_counts(normalize = True)) 
print('\n')
print(df.head(10))

#heatmap shows pearson's correlation of all features with default
fig, ax = plt.subplots(figsize=(15,10)) 
cm_df = sns.heatmap(df.corr(),annot=True, fmt = ".2f", cmap = "coolwarm", ax=ax)
plt.show()

#export ready data to csv
df.to_csv('data_ready_for_ML_.csv')

