#!/usr/bin/env python
# coding: utf-8

# # The data is related with direct marketing campaigns of a Portuguese banking institution.The marketing campaigns were based on phone calls.Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
# 
# 

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.feature_extraction.text import CountVectorizer  #DT does not take strings as input for the model fit step....
from IPython.display import Image  
#import pydotplus as pydot
from sklearn import tree
from os import system



# ### Read and Review data 

# In[2]:


campdata = pd.read_csv("bank-full.csv")
     


# In[3]:


campdata.shape


# ###  There are 45,211 rows in the Campaign data set with 17 columns

# ### Check the data types of all attributes

# In[4]:


campdata.info()


# ### Many attributes are of object type which needs to handled when working with decision trees and ensemble models

# In[5]:


campdata.isnull().sum()


# ###  No null values are populatd in data

# In[6]:


# Five point summary of  the data set

campdata.describe().transpose()


# In[7]:


print(campdata.job.value_counts())
print("---------------------------")
print(campdata.marital.value_counts())
print("---------------------------")
print(campdata.education.value_counts())

print("---------------------------")
print(campdata.default.value_counts())

print("---------------------------")
print(campdata.loan.value_counts())

print("---------------------------")
print(campdata.contact.value_counts())

print("---------------------------")
print(campdata.month.value_counts())

print("---------------------------")
print(campdata.poutcome.value_counts())


print("---------------------------")
print(campdata.Target.value_counts())


# ###  Above result shows the break up of various values for each columns.This is very important to understand the multi colinearity of the each attribute

# In[8]:


campdata.skew(axis = 0, skipna = True)


# In[9]:


boxplot1 = campdata.boxplot(["age" ,"campaign","pdays","previous"] )


# ###  The above box plot shows no may outliers exists in numeric attributes 

# In[10]:


for feature in campdata.columns: # Loop through all columns in the dataframe
    if campdata[feature].dtype == 'object': # Only apply for columns with categorical strings
        campdata[feature] = pd.Categorical(campdata[feature])# Replace strings with an integer
campdata.head(10)


# In[11]:


campdata.info()


# ### Convert all the Category variables to Numberic.This could be achieved by creating Dummy columns for each categorical variable , identify the multi colinearity and then drop the last dummy column
# 
# For E.g : If columns x , y & z are multi colinear then can drop coumn z
# 
# Also append all the dummy coloumns to campdata then drop actualy coloumns to avoid duplication.
# 
# 

# In[12]:


dummies_job = pd.get_dummies(campdata.job)
dummies_job.drop('unknown',axis='columns',inplace=True)


dummies_marital =  pd.get_dummies(campdata.marital)


dummies_education =  pd.get_dummies(campdata.education)
dummies_education.drop('unknown',axis='columns',inplace=True)



dummies_default =    pd.get_dummies(campdata.default)
dummies_default.rename(columns = {'yes':'default_yes'}, inplace = True)
dummies_default.drop("no",axis='columns',inplace=True)



dummies_housing =    pd.get_dummies(campdata.housing)
dummies_housing.rename(columns = {'yes':'housing_yes'}, inplace = True)
dummies_housing.drop("no",axis='columns',inplace=True)



dummies_loan = pd.get_dummies(campdata.loan)
dummies_loan.rename(columns= {'yes':'loans_yes'},inplace = True)
dummies_loan.drop("no",axis='columns',inplace=True)


dummies_contact =    pd.get_dummies(campdata.contact)
dummies_contact.drop("unknown",axis='columns',inplace = True)


dummies_month =      pd.get_dummies(campdata.month)
dummies_month.drop("dec",axis='columns',inplace=True)


dummies_poutcome =    pd.get_dummies(campdata.poutcome)
dummies_poutcome.rename(columns = {'failure':'Prev_Fail','success' : 'Prev_success',
                                 'other' : 'prev_other','unknown' :'prev_unknown'}  , inplace = True)


dummies_poutcome.drop("prev_unknown",axis='columns',inplace=True)

dummies_Target  =    pd.get_dummies(campdata.Target)       
dummies_Target.rename(columns= {'yes':'Target_yes'},inplace = True)
dummies_Target.drop("no",axis='columns',inplace=True)


frames = [campdata ,dummies_job,dummies_marital,dummies_education,dummies_default,dummies_housing,dummies_loan,dummies_contact,
          
          dummies_month,dummies_poutcome,dummies_Target]

datafix = pd.concat(frames,axis = 'columns')


datafix.drop(["age","job",'marital','education','default','housing','loan','contact','month','poutcome','Target'],axis = 'columns' ,inplace=True)

Final_data = datafix


# In[13]:



Final_data.info()


# ###  Converting the caterogical variable in to numberic is complete.Now, all the data is of numeric type so that we can apply various ensemble models

# In[14]:


Final_data.describe()


# ## Split Data

# ###  The predection variable is Target_yes coloumn 

# In[15]:


X = Final_data.drop("Target_yes" , axis=1)
y = Final_data.pop("Target_yes")


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)


# In[17]:


X_train


# ## Build Decision Tree Model

# We will build our model using the DecisionTreeClassifier function. Using default 'gini' criteria to split. Other option include 'entropy'.  

# In[18]:


dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(X_train, y_train)


# ### Try decision tree with gini impurity 

# ## Scoring our Decision Tree

# In[74]:


print(dTree.score(X_train, y_train))
print(dTree.score(X_test, y_test))


# ### The above result indicates over fitting and requires fine tuning

# ## Reducing over fitting (Regularization)

# In[75]:


dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state=1)
dTreeR.fit(X_train, y_train)
print(dTreeR.score(X_train, y_train))
print(dTreeR.score(X_test, y_test))


# ###  Post adjusting max_depth = 3  - Train score  = 90 %   Test score = 90 % which is a good indicator

# In[76]:


# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print (pd.DataFrame(dTreeR.feature_importances_, columns = ["Imp"], index = X_train.columns))


# ###  Feature : Duration is highest contribute to the predection followed by Prev_success 

# In[77]:


print(dTreeR.score(X_test , y_test))
y_predict = dTreeR.predict(X_test)

cm=metrics.confusion_matrix(y_test, y_predict, labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# #                             Ensemble Learning - Bagging

# In[22]:


from sklearn.ensemble import BaggingClassifier

bgcl = BaggingClassifier(base_estimator=dTree, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(X_train, y_train)


# In[23]:


y_predict = bgcl.predict(X_test)

print(bgcl.score(X_test , y_test))

cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# ### Ensemble Boosting delivers 90.5 % of score too which is almost same as Decision tree

# # Ensemble Learning - AdaBoosting

# In[24]:


from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(n_estimators=10, random_state=1)
#abcl = AdaBoostClassifier( n_estimators=50,random_state=1)
abcl = abcl.fit(X_train, y_train)


# In[81]:


y_predict = abcl.predict(X_test)
print(abcl.score(X_test , y_test))

cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# ### Ada Boosting delivers 89.3 % of score

# #                     Ensemble Learning - GradientBoost

# In[82]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 50,random_state=1)
gbcl = gbcl.fit(X_train, y_train)


# In[83]:


y_predict = gbcl.predict(X_test)
print(gbcl.score(X_test, y_test))
cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# ### Gradient Bossting too delivers simimar score 90.5 % 

# # Ensemble RandomForest Classifier

# In[84]:


from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators = 50, random_state=1,max_features=12)
rfcl = rfcl.fit(X_train, y_train)


# In[85]:


y_predict = rfcl.predict(X_test)
print(rfcl.score(X_test, y_test))
cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# ### Conclusion :  Random tree calssified score marginally higher with 90.8 % - seems better of all above
