#!/usr/bin/env python
# coding: utf-8

# In[67]:


#Import all Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


#Read Test and Train Data
salarytest = pd.read_csv(r'C:\Users\Rohit\Downloads\tcdml1920-income-ind\without labels.csv')
salary = pd.read_csv(r'C:\Users\Rohit\Downloads\tcdml1920-income-ind\tcd ml 2019-20 income prediction training (with labels).csv')


# In[69]:


#Drop Instance and Income column.
salarytest.drop(['Instance', 'Income'],1,inplace = True)
salary.drop(['Instance'],1,inplace = True)


# In[70]:


#Dealing with Missing Values
salary['Gender'] = salary['Gender'].fillna('unknown')
salary['Gender'] = salary['Gender'].replace('0','unknown')
salary['Gender'] = salary['Gender'].replace('other','Other')
salarytest['Gender'] = salarytest['Gender'].fillna('unknown')
salarytest['Gender'] = salarytest['Gender'].replace('0','unknown')
salarytest['Gender'] = salarytest['Gender'].replace('other','Other')
#salarytest['Gender'] = salarytest['Gender'].replace('0',np.NaN)
salary['University Degree'] = salary['University Degree'].fillna('No')
salary['University Degree'] = salary['University Degree'].replace('0','No')
salarytest['University Degree'] = salarytest['University Degree'].fillna('N0')
salarytest['University Degree'] = salarytest['University Degree'].replace('0','No')
#salarytest['University Degree'] = salarytest['University Degree'].replace('0',np.NaN)
salary['Hair Color'] = salary['Hair Color'].fillna('other')
salary['Hair Color'] = salary['Hair Color'].replace('0','other')
salary['Hair Color'] = salary['Hair Color'].replace('Unknown','other')
salarytest['Hair Color'] = salarytest['Hair Color'].fillna('other')
salarytest['Hair Color'] = salarytest['Hair Color'].replace('0','other')
salarytest['Hair Color'] = salarytest['Hair Color'].replace('Unknown','other')
#salarytest['Hair Color'] = salarytest['Hair Color'].replace('0',np.NaN)
#salary['Year of Record'] = salary['Year of Record'].replace(np.nan,0)
#salary['Age'] = salary['Age'].replace(np.nan,0)
salary['Year of Record'] = salary['Year of Record'].fillna(salary['Year of Record'].median())
salary['Age'] = salary['Age'].fillna(salary['Age'].median())
salary['Profession'] = salary['Profession'].fillna('other')
salarytest['Year of Record'] = salarytest['Year of Record'].fillna(np.mean(salarytest['Year of Record']))
#salarytest['Age'] = salarytest['Age'].replace(np.nan,0)
#salarytest['Year of Record'] = salarytest['Year of Record'].replace(0,np.median(salary['Year of Record']))
salarytest['Age'] = salarytest['Age'].fillna(np.mean(salarytest['Age']))
salarytest['Profession'] = salarytest['Profession'].fillna('other')
#salary['Country'] = salary['Country'].replace('0',np.NaN)
#salarytest['Country'] = salarytest['Country'].replace('0',np.NaN)
#salary['Country'] = salary['Country'].replace('Unknown',np.NaN)
#salarytest['Country'] = salarytest['Country'].replace('Unknown',np.NaN)


# In[71]:


#get Dummies for Gender
gender = pd.get_dummies(salary['Gender'],drop_first = True)
salary = pd.concat([salary,gender],axis = 1)
salary.drop(['Gender'], axis = 1, inplace = True)
gendertest = pd.get_dummies(salarytest['Gender'],drop_first = True)
salarytest = pd.concat([salarytest,gendertest],axis = 1)
salarytest.drop(['Gender'], axis = 1, inplace = True)


# In[72]:


#get Dummies for Hair Color
hair = pd.get_dummies(salary['Hair Color'],drop_first = True)
salary = pd.concat([salary,hair],axis = 1)
salary.drop(['Hair Color'],axis = 1,inplace = True)
hairtest = pd.get_dummies(salarytest['Hair Color'],drop_first = True)
salarytest = pd.concat([salarytest,hairtest],axis = 1)
salarytest.drop(['Hair Color'],axis = 1,inplace = True)


# In[73]:


#get Dummies for University Degree
deg = pd.get_dummies(salary['University Degree'],drop_first = True)
salary = pd.concat([salary,deg],axis = 1)
salary.drop(['University Degree'],axis = 1,inplace = True)
#salary['University Degree'] = salary['University Degree'].replace('Master',2)
#salary['University Degree'] = salary['University Degree'].replace('Bachelor',1)
#salary['University Degree'] = salary['University Degree'].replace('No',0)
#salary['University Degree'] = salary['University Degree'].replace('PhD',3)
degtest = pd.get_dummies(salarytest['University Degree'], drop_first = True)
salarytest = pd.concat([salarytest,degtest],axis = 1)
salarytest.drop(['University Degree'],axis = 1,inplace = True)
#salarytest['University Degree'] = salarytest['University Degree'].replace('Master',2)
#salarytest['University Degree'] = salarytest['University Degree'].replace('Bachelor',1)
#salarytest['University Degree'] = salarytest['University Degree'].replace('No',0)
#salarytest['University Degree'] = salarytest['University Degree'].replace('PhD',3)


# In[74]:


country = salary['Country'].value_counts()
countrytest = salarytest['Country'].value_counts()


# In[75]:


profession = salary['Profession'].value_counts()
professiontest = salarytest['Profession'].value_counts()


# In[76]:


salary['newcountry'] = 0
salary['newprofession'] = 0
salarytest['newcountry'] = 0
salarytest['newprofession'] = 0


# In[77]:


#Response encoding for Country
for i in country.index:
    temp = salary[salary['Country']==i]
    #a = sum(temp['Income in EUR'])/temp.shape[0]
    salary.loc[salary['Country']==i, 'newcountry'] = ((np.mean(temp['Income in EUR']) * len(temp)) +(10 * np.mean(salary['Income in EUR'])))/(len(temp)+10)
    salarytest.loc[salarytest['Country']==i,'newcountry'] = ((np.mean(temp['Income in EUR']) * len(temp)) +(10 * np.mean(salary['Income in EUR'])))/(len(temp)+10)
    
    
    
    
    
    
#for i in countrytest.index:
    #salarytest['newcountry'][salarytest['Country']==i] = salary['newcountry'][salary['Country']==i].values[1]


# In[78]:


#Response endcoding for Profession
for i in profession.index:
    temp = salary[salary['Profession']==i]
    #a = sum(temp['Income in EUR'])/temp.shape[0]
    salary.loc[salary['Profession']==i,'newprofession'] = ((np.mean(temp['Income in EUR']) * len(temp)) + (10 * np.mean(salary['Income in EUR'])))/(len(temp)+10)
    salarytest.loc[salarytest['Profession']==i,'newprofession'] = ((np.mean(temp['Income in EUR']) * len(temp)) + (10 * np.mean(salary['Income in EUR'])))/(len(temp)+10)
#salarytest['newprofession'] = salary['newprofession']


# In[79]:


salary.drop(['Country','Profession'],axis = 1, inplace = True)
salarytest.drop(['Country','Profession'],axis = 1, inplace = True)


# In[80]:


salarytest['newcountry'] = salarytest['newcountry'].replace(0,np.mean(salarytest['newcountry']))
salarytest['newprofession'] = salarytest['newprofession'].replace(0,np.mean(salarytest['newprofession']))


# In[81]:


from sklearn.model_selection import train_test_split


# In[82]:


Y = salary['Income in EUR']
X = salary.drop(['Income in EUR', 'Blond','Brown', 'Red', 'other','Wears Glasses'],axis = 1)
Xtest = salarytest.drop(['Blond','Brown','Red','other','Wears Glasses'], axis  = 1)


# In[ ]:


#function for scaling
def normalize(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))


# In[ ]:


X = X.apply(normalize)
Xtest = Xtest.apply(normalize)


# In[85]:


X_train, X_test,Y_train, Y_test = train_test_split(X,Y,train_size = 0.9)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor(n_estimators = 700)


# In[ ]:


rf.fit(X_train,Y_train)


# In[ ]:


ypred = rf.predict(X_test)


# In[90]:


from sklearn.metrics import mean_squared_error


# In[91]:


mse = mean_squared_error(Y_test,ypred)


# In[92]:


import math


# In[93]:


math.sqrt(mse)


# In[ ]:


ypred = rf.predict(Xtest)


# In[ ]:


ypred = pd.DataFrame(ypred)


# In[ ]:


ypred.to_csv(r'C:\Users\Rohit\Downloads\tcdml1920-income-ind\result10.csv')


# In[ ]:


#Tried Hyperparameter tuning but getting memory allocation error
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
#max_features = ['auto', 'sqrt']
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
#min_samples_split = [2, 5, 10]
#min_samples_leaf = [1, 2, 4]
#bootstrap = [True, False]
#random_grid = {'n_estimators': n_estimators,
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}


# In[ ]:


#from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 4, verbose=2, random_state=42, n_jobs = 4)


# In[ ]:


#rf_random.fit(X,Y)

