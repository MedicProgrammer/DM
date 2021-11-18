#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
X_train = pd.read_csv('C:/Users/SAMSUNG/Downloads/train_input.csv')
X_test = pd.read_csv('C:/Users/SAMSUNG/Downloads/test_input.csv')
y_train = pd.read_csv('C:/Users/SAMSUNG/Downloads/train_output.csv')
answer = pd.read_csv('C:/Users/SAMSUNG/Downloads/test_output_sample.csv')
X_train


# In[2]:


# relace missing values into mean value


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train)
X_train = pd.DataFrame(data=imp.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(data = imp.transform(X_test), columns = X_test.columns)


# In[3]:


# check for outliers(Z>4) and replace it to the mean value

def Outliers(data):
    threshold=4
    mean = np.mean(data)
    std = np.std(data)
    z = [(y-mean)/std for y in data]
    return np.where(np.abs(z)>threshold)

def Outliers_for_test(col):
    threshold=4
    mean = np.mean(X_train[col])
    std = np.std(X_train[col])
    z = [(y-mean)/std for y in X_test[col]]
    return np.where(np.abs(z)>threshold)

for col in X_train.columns:
    if(np.std(X_train[col])>0.0001):
        sum_of_outliers = np.sum(X_train[col][Outliers(X_train[col])[0]])
        mean = (np.sum(X_train[col]) - sum_of_outliers) / (len(X_train[col]) - len(Outliers(X_train[col])[0]))
        X_train[col][Outliers(X_train[col])[0]] = mean

for col in X_test.columns:
    if(np.std(X_train[col])>0.0001):
        sum_of_outliers = np.sum(X_test[col][Outliers_for_test(col)[0]])
        mean = (np.sum(X_test[col]) - sum_of_outliers) / (len(X_test[col]) - len(Outliers_for_test(col)[0]))
        X_test[col][Outliers_for_test(col)[0]] = mean


# In[4]:


# add some data(target=1), just copy&paste for 4 times

semiconductor = X_train
semiconductor['target'] = y_train


for i in range(0,3):
    semiconductor = semiconductor.append(semiconductor.loc[semiconductor['target']==1]).sample(frac=1.0)


# In[5]:


# Conduct Logistic Regrssion, find feature importances, and choose the important features

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='lbfgs',max_iter=20000,C=2.0)
)
y_train = semiconductor['target']
X_train = semiconductor.drop(['target'],axis=1)
model.fit(X_train, y_train)
importance_ranking = pd.DataFrame(model[1].coef_[0]).rank(ascending=False)

important_features = []
for i,col in enumerate(X_train.columns):
    importance = model[1].coef_[0][i]
    if (importance_ranking.iloc[i][0]) > 20:
        semiconductor = semiconductor.drop(col,axis=1)
        X_test = X_test.drop(col,axis=1)
    else:
        important_features.append(col)


# In[6]:


# Conduct LogisticRegression again, only with the important variables

for i in range(0,100):
    selected_important_features = random.sample(important_features, 15)

    X_tr = X_train[selected_important_features]
    X_te = X_test[selected_important_features]
    #semiconductor
    #print(len(semiconductor.columns), np.max(model[1].coef_[0]))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='lbfgs',max_iter=20000,C=2.0)
    )
    model.fit(X_tr,y_train)

    answer['label'] = model.predict(X_te)
    #print(cross_val_score(model, X_train, y_train, cv=10))
    answer.to_csv(str(str(i)+str(selected_important_features)+'.csv'),index=False)


# In[7]:


X_train.describe()


# In[8]:


model[1].coef_[0]


# In[9]:


semiconductor.columns


# In[10]:


#C=2.5, importance<0.955 : score = 0.5
#(상위 15개)

