#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cpt as cpt
from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import LinearSVC as svc


# In[2]:


#import data
train = np.loadtxt(open("data/data/train.csv", "rb"), delimiter=",")
test = np.loadtxt(open("data/data/test.csv", "rb"), delimiter=",")

#split data
x_train = train[:,1:201]
y_train = train[:,0]
x_test = test[:,1:201]
y_test = test[:,0]

#Convert 0 to -1 for cpt
y_train[y_train==0] = -1
y_test[y_test==0] = -1

#soft margin 
soft = 1


# In[3]:


#base Parent class
class svm:
    #finds opt line 
    #data = training, ans = labbles, C = soft margin
    def fit(self, data, ans, C):
        return
    
    #finds projection
    def project(self, data):
        return np.dot(data, self.w) + self.b
    
    #makes prediction
    def predict(self, data):
        return np.sign(self.project(data))


# In[62]:


#primal problem
class SVM_prime(svm):
    def fit(self, data, ans, C):
        
        n_samples, n_features = data.shape
        n_sum = n_samples + n_features
        
        # P = Diagnal Matix
        P = cpt.matrix(np.eye(n_sum+1))
        
        # q = 0(n_samples), C(n_samples)
        q = cpt.matrix(np.vstack([np.zeros((n_features+1,1)), C*np.ones((n_samples, 1))]))
        
        # A = -G^Tz
        A = np.zeros((2*n_samples, n_sum+1))
        A[:n_samples,0:n_features] = np.dot(ans,data)
        A[:n_samples,n_features] = ans.T
        A[:n_samples,n_features+1:]  = np.eye(n_samples)
        A[n_samples:,n_features+1:] = np.eye(n_samples)
        A = cpt.matrix(-A)                  
        
        # b = Ax
        b = np.zeros((2*n_samples,1))
        b[:n_samples] = -1
        b = cpt.matrix(b)
        
        # solver
        solv = cpt.solvers.qp(P, q, A, b)
 
        # Bias
        self.b = solv['x'][n_features]
        
        # Weights
        self.w = np.array(solv['x'][:n_features])
        


# In[5]:


#dual problem
class SVM_dual(svm):
    def fit(self, data, ans, C):
        
        n_samples, n_features = data.shape
        
        # P = (data^T)data
        temp = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for k in range(n_samples):
                temp[i,k] = np.dot(data[i], data[k])
        P = cpt.matrix(np.outer(ans, ans) * temp)
        
        # q = -1(1*n_samples)
        q = cpt.matrix(np.ones(n_samples) * -1)
        
        # G = -1(n_samples*n_samples)
        G = cpt.matrix(np.vstack((np.eye(n_samples)*-1,np.eye(n_samples))))
        
        # h = 0(1*n_samples), C(1*n_samples)
        h = cpt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
        
        # A = ans^T 
        A = cpt.matrix(ans, (1, n_samples))

        # b = 0 
        b = cpt.matrix(0.0)
        
        # solver
        solv = cpt.solvers.qp(P, q, G, h, A, b)
        
        # remove all zero/neg values
        alpha = np.ravel(solv['x'])
        self.a = alpha[alpha > 1e-6]
        self.b = 0
        self.w = np.zeros(n_features)
        
        xx = data[alpha > 1e-6]
        yy = ans[alpha > 1e-6]
        aa = np.arange(len(alpha))[alpha > 1e-6]
        bb = len(self.a)
    

        # Bais and weight
        for i in range(bb):
            self.b += yy[i]
            self.b -= np.sum(self.a * yy * temp[aa[i], alpha > 1e-6])
            self.w += self.a[i] *  xx[i] * yy[i] 
        
        self.b = self.b / bb
        


# In[63]:


#svm prime
svmP = SVM_prime()
svmP.fit(x_train, y_train, soft)


# In[7]:


#svm dual 
svmD = SVM_dual()
svmD.fit(x_train, y_train, soft)


# In[8]:


#output prime results
test1 = svmP.predict(x_test)
res1 = cm(y_test, test1)
print(res1)
print(svmP.b)
print(svmP.w)


# In[9]:


#output prime results
test2 = svmD.predict(x_test)
res2 = cm(y_test, test2)
print(res2)
print(svmD.b)
print(svmD.w)
print(svmD.a)


# In[10]:


#lib prime
libP = svc()
libP.set_params(dual=False)
libP.set_params(C=soft)
#svc.set_params(max_iter=10000)
libP.fit(x_train, y_train)
test3 = libP.predict(x_test)
print(cm(y_test, test3))
print(libP.coef_)


# In[50]:


#lib dual
libD = svc()
libD.set_params(dual=True)
libD.set_params(C=soft)
libD.set_params(max_iter=100000)
libD.fit(x_train, y_train)
test4 = libD.predict(x_test)
print(cm(y_test, test4))
print(libD.coef_)
print(libD.intercept_)


# In[ ]:




