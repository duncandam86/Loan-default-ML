# import packages and libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler


#Using panda to read csv file into a dataframe
df = pd.read_csv('data_ready_for_ML_.csv', index_col = 'id')
print(df.shape)
print(df.info())
print(df.head())

#1) SELECT DATA
#select dependent and independent data
X = df.drop('default', axis = 1)
y = df['default']

#Scaling X features
X_transformed = RobustScaler().fit_transform(X)

#2) BALANCE DATA USING SMOTE
#initiate SMOTE
smt = SMOTE (random_state = 42)
X_resampled, y_resampled = smt.fit_resample(X_transformed, y)
#checking whether resampling works
unique_elements, counts_elements = np.unique(y_resampled, return_counts=True)


#3) CREATE TRAIN AND TEST DATA SETS
X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled, test_size=.3, random_state=42)

#4) MODEL EVALUATION
def evaluate_model(model, y_test, y_pred, y_prob):
    a,b,c,d = model, accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob), f1_score(y_test, y_pred, average='weighted')
    score_table = {'model': a, 'accuracy_score': b, 'auc_score': c, 'f1_score': d}
    df_score_table = pd.DataFrame(data = score_table, index = [1])
    print('\n ---Score table for ' + model + '---')
    print(df_score_table)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('\n ---Confusion matrix for ' + model + '---')
    print({'tn':tn,'fp':fp,'fn':fn,'tp':tp})

#initiate models
lg = LogisticRegression(random_state = 42, solver = 'lbfgs')
tree = DecisionTreeClassifier(random_state = 42)
rfc = RandomForestClassifier(random_state = 42, n_estimators = 10)
gbc = GradientBoostingClassifier(random_state = 42)
abc = AdaBoostClassifier(random_state = 42)
xgb = XGBClassifier(objective="binary:logistic", random_state=42)

# fit model with logistic regression without tuning hyperparameters
lg.fit(X_train, y_train)
lg_y_pred = lg.predict(X_test)
lg_y_prob = lg.predict_proba(X_test)[:,1]
#show evaluation score
evaluate_model('Logistic Regression',y_test, lg_y_pred, lg_y_prob)

#fit model with decision tree without tuning hyperparameters 
tree.fit(X_train, y_train)
tree_y_pred = tree.predict(X_test)
tree_y_prob = tree.predict_proba(X_test)[:,1]
#show evaluation score
evaluate_model('Decision Tree',y_test, tree_y_pred, tree_y_prob)

#fit model with Random forest classifier without  tuning hyperparameters
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)
rfc_y_prob = rfc.predict_proba(X_test)[:,1]
#show evaluation score
evaluate_model('Random Forest Classifier',y_test, rfc_y_pred, rfc_y_prob)

#fit model with Gradient Boosting Classifier without tuning hyperparameters 
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)
gbc_y_prob = gbc.predict_proba(X_test)[:,1]
#show evaluation score
evaluate_model('Gradient Boosting Classifier',y_test, gbc_y_pred, gbc_y_prob)

#fit model with Adaptive Boosting Classifier without tuning hyperparameters 
abc.fit(X_train, y_train)
abc_y_pred = abc.predict(X_test)
abc_y_prob = abc.predict_proba(X_test)[:,1]
#show evaluation score
evaluate_model('Apdative Boosting Classifier',y_test, abc_y_pred, abc_y_prob)

#fit model with XGB Classifier without tuning hyperparameters 
xgb.fit(X_train, y_train)
xgb_y_pred = xgb.predict(X_test)
xgb_y_prob = xgb.predict_proba(X_test)[:,1]
#show evaluation score
evaluate_model('XGB Classifier',y_test, xgb_y_pred, xgb_y_prob)

