#!/usr/bin/env python
# coding: utf-8

# In[1]:


#sklearn
import numpy as np
import cvxopt
import csv
from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import LinearSVC as svc
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics


# In[2]:


#import data
col = range(2,32)
data = np.loadtxt(open("data.csv", "rb"),usecols=col  , delimiter="," )
data2 = np.loadtxt(open("data.csv", "rb"),usecols=1 , dtype='<S1', delimiter="," )


data3 = list()

for i in data2:
    if(i==b'B'):
        data3.append(1)
    else:
        data3.append(-1)

        
#split data
x_train = data[:300]
y_train = data3[:300]
x_test = data[300:]
y_test = data3[300:]


# In[3]:


#svc=SVC(probability=True, kernel='linear')
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model = abc.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Score:",metrics.accuracy_score(y_test, y_pred))

