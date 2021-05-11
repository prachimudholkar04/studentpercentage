#!/usr/bin/env python
# coding: utf-8

# # Importing all libraries

# In[4]:


import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading dataset from file
# 

# In[5]:


path =  r'C:\Users\User\Desktop\StudentsPerformance (1).csv'
student_data = pd.read_csv(path)
print("Data imported successfully")

student_data.head(10)


# # Descerpition of data

# In[97]:


student_data.describe()


# # Reading last data

# In[98]:


student_data.tail(10)


# In[99]:


student_data.info()


# In[100]:


student_data.shape


# # Plotting distribution plot

# In[101]:


student_data.plot(x='HOURS', y='SCORES', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hour Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Preaparing Data

# In[102]:


X = student_data.iloc[:, :-1].values  
y = student_data.iloc[:, 1].values  


# # Divide data into two sets.i.e Traning Set and Test Set

# In[103]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[104]:


sns.distplot(y_train, kde=True)
plt.title("DISTRIBUTION OF SCORES")
plt.xlabel('HOURS STUIDED')
plt.ylabel('PERCENTAGE SCORED')


# In[105]:


sns.regplot(X_train, y_train)
plt.title('HOURS VS SCORES')
plt.xlabel('HOURS STUDIED')
plt.ylabel('PERCENTAGE SCORED')


# # Train the algorithm

# In[106]:


from sklearn.linear_model import LinearRegression  
train = LinearRegression()  
train.fit(X_train, y_train) 
print("Training complete by "+str(train)) 


# In[107]:


# Plotting the regression line
line = train.coef_*X+train.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[108]:


#predicting test results
y_pred = train.predict(X_test)

#Comparing actual vs predict
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head()


# # Visulizaing Results

# In[109]:


#visualizing the result
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, train.predict (X_train), color = 'red')
plt.title('Training set Graph')
plt.xlabel('HOURS')
plt.ylabel('PERCENTAGE')


# # Using own data to predict percentage

# In[110]:


# You can also test with your own data
HOURS = 12
Prediction = train.predict([[HOURS]])
print("No of Hours = {}".format(HOURS))
print("Predicted Score = {}".format(Prediction[0]))


# # Evaluating Model

# In[111]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:





# In[ ]:





# In[ ]:




