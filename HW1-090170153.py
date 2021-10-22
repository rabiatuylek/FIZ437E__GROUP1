#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

M = 10 #degree of polynomial

data = 5
model = np.linspace(0,1,num=data)
train_data = np.linspace(0,1,num=data)
test_data = np.linspace(0,1,num=data)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, test_data, test_size = 0.20)

f1_model = np.sin(2*np.pi*model)
f1_train = np.sin(2*np.pi*X_train)
f1_test  = np.sin(2*np.pi*X_test)
#len(X_train) = 4
#len(X_test) = 1
gaussian_model = np.random.normal(0,0.05,data)
gaussian_train = np.random.normal(0,0.05,len(X_train))
gaussian_test = np.random.normal(0,0.05,len(X_test))

t_model = f1_model + gaussian_model
train_model = f1_train + gaussian_train
test_model = f1_test + gaussian_test

# fitting the 10th polynomial 
p = np.polyfit(model,f1_model,M)
train_over_fitting =np.polyfit(X_train, train_model,M)
test_over_fitting  =np.polyfit(X_test, test_model,M)


#create polynomial in numpy = np.poly1d
model_polynomial =np.poly1d(p)
train_polynomial =np.poly1d(train_over_fitting)
test_polynomial =np.poly1d(test_over_fitting)

#outputs
model_final = model_polynomial(model)
y_train =  train_polynomial(Y_train)
y_test = test_polynomial(Y_test)


#show in graphıcs
plt.figure
plt.title("5 Data")
plt.plot(X_train,train_model,'go',model,model_final,'r-')
plt.plot(X_test,test_model,'bo')
plt.legend(('Training Data','Model','Testing Data'))

# MSE Loss Function
L_2 = 0.5*(np.sum(y_train-train_model)**2)
print("Training Error:",L_2)  # Training Error

# MSE Loss Function
L_2_test = 0.5*(np.sum(y_test-test_model)**2)
print("Testing Error:",L_2_test)  # Testing Error


# In[15]:


M = 10 #degree of polynomial

data = 10
model = np.linspace(0,1,num=data)
train_data = np.linspace(0,1,num=data)
test_data = np.linspace(0,1,num=data)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, test_data, test_size = 0.20)

f1_model = np.sin(2*np.pi*model)
f1_train = np.sin(2*np.pi*X_train)
f1_test  = np.sin(2*np.pi*X_test)
#len(X_train) = 8
#len(X_test) = 2
gaussian_model = np.random.normal(0,0.05,data)
gaussian_train = np.random.normal(0,0.05,len(X_train))
gaussian_test = np.random.normal(0,0.05,len(X_test))

t_model = f1_model + gaussian_model
train_model = f1_train + gaussian_train
test_model = f1_test + gaussian_test

# fitting the 10th polynomial 
p = np.polyfit(model,f1_model,M)
train_over_fitting =np.polyfit(X_train, train_model,M)
test_over_fitting  =np.polyfit(X_test, test_model,M)


#create polynomial in numpy = np.poly1d
model_polynomial =np.poly1d(p)
train_polynomial =np.poly1d(train_over_fitting)
test_polynomial =np.poly1d(test_over_fitting)

#outputs
model_final = model_polynomial(model)
y_train =  train_polynomial(Y_train)
y_test = test_polynomial(Y_test)


#show in graphıcs
plt.figure
plt.title("10 Data")
plt.plot(X_train,train_model,'go',model,model_final,'r-')
plt.plot(X_test,test_model,'bo')
plt.legend(('Training Data','Model','Testing Data'))

# MSE Loss Function
L_2 = 0.5*(np.sum(y_train-train_model)**2)
print("Training Error:",L_2)  # Training Error

# MSE Loss Function
L_2_test = 0.5*(np.sum(y_test-test_model)**2)
print("Testing Error:",L_2_test)  # Testing Error


# In[18]:


M = 10 #degree of polynomial

data = 20
model = np.linspace(0,1,num=data)
train_data = np.linspace(0,1,num=data)
test_data = np.linspace(0,1,num=data)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, test_data, test_size = 0.20)

f1_model = np.sin(2*np.pi*model)
f1_train = np.sin(2*np.pi*X_train)
f1_test  = np.sin(2*np.pi*X_test)
#len(X_train) = 16
#len(X_test) = 4
gaussian_model = np.random.normal(0,0.05,data)
gaussian_train = np.random.normal(0,0.05,len(X_train))
gaussian_test = np.random.normal(0,0.05,len(X_test))

t_model = f1_model + gaussian_model
train_model = f1_train + gaussian_train
test_model = f1_test + gaussian_test

# fitting the 10th polynomial 
p = np.polyfit(model,f1_model,M)
train_over_fitting =np.polyfit(X_train, train_model,M)
test_over_fitting  =np.polyfit(X_test, test_model,M)


#create polynomial in numpy = np.poly1d
model_polynomial =np.poly1d(p)
train_polynomial =np.poly1d(train_over_fitting)
test_polynomial =np.poly1d(test_over_fitting)

#outputs
model_final = model_polynomial(model)
y_train =  train_polynomial(Y_train)
y_test = test_polynomial(Y_test)


#show in graphıcs
plt.figure
plt.title("20 Data")
plt.plot(X_train,train_model,'go',model,model_final,'r-')
plt.plot(X_test,test_model,'bo')
plt.legend(('Training Data','Model','Testing Data'))

# MSE Loss Function
L_2 = 0.5*(np.sum(y_train-train_model)**2)
print("Training Error:",L_2)  # Training Error

# MSE Loss Function
L_2_test = 0.5*(np.sum(y_test-test_model)**2)
print("Testing Error:",L_2_test)  # Testing Error


# In[19]:


M = 10 #degree of polynomial

data = 50
model = np.linspace(0,1,num=data)
train_data = np.linspace(0,1,num=data)
test_data = np.linspace(0,1,num=data)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, test_data, test_size = 0.20)

f1_model = np.sin(2*np.pi*model)
f1_train = np.sin(2*np.pi*X_train)
f1_test  = np.sin(2*np.pi*X_test)
#len(X_train) = 40
#len(X_test) = 10
gaussian_model = np.random.normal(0,0.05,data)
gaussian_train = np.random.normal(0,0.05,len(X_train))
gaussian_test = np.random.normal(0,0.05,len(X_test))

t_model = f1_model + gaussian_model
train_model = f1_train + gaussian_train
test_model = f1_test + gaussian_test

# fitting the 10th polynomial 
p = np.polyfit(model,f1_model,M)
train_over_fitting =np.polyfit(X_train, train_model,M)
test_over_fitting  =np.polyfit(X_test, test_model,M)


#create polynomial in numpy = np.poly1d
model_polynomial =np.poly1d(p)
train_polynomial =np.poly1d(train_over_fitting)
test_polynomial =np.poly1d(test_over_fitting)

#outputs
model_final = model_polynomial(model)
y_train =  train_polynomial(Y_train)
y_test = test_polynomial(Y_test)


#show in graphıcs
plt.figure
plt.title("50 Data")
plt.plot(X_train,train_model,'go',model,model_final,'r-')
plt.plot(X_test,test_model,'bo')
plt.legend(('Training Data','Model','Testing Data'))

# MSE Loss Function
L_2 = 0.5*(np.sum(y_train-train_model)**2)
print("Training Error:",L_2)  # Training Error

# MSE Loss Function
L_2_test = 0.5*(np.sum(y_test-test_model)**2)
print("Testing Error:",L_2_test)  # Testing Error


# In[20]:


M = 10 #degree of polynomial

data = 100
model = np.linspace(0,1,num=data)
train_data = np.linspace(0,1,num=data)
test_data = np.linspace(0,1,num=data)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, test_data, test_size = 0.20)

f1_model = np.sin(2*np.pi*model)
f1_train = np.sin(2*np.pi*X_train)
f1_test  = np.sin(2*np.pi*X_test)
#len(X_train) = 80
#len(X_test) = 20
gaussian_model = np.random.normal(0,0.05,data)
gaussian_train = np.random.normal(0,0.05,len(X_train))
gaussian_test = np.random.normal(0,0.05,len(X_test))

t_model = f1_model + gaussian_model
train_model = f1_train + gaussian_train
test_model = f1_test + gaussian_test

# fitting the 10th polynomial 
p = np.polyfit(model,f1_model,M)
train_over_fitting =np.polyfit(X_train, train_model,M)
test_over_fitting  =np.polyfit(X_test, test_model,M)


#create polynomial in numpy = np.poly1d
model_polynomial =np.poly1d(p)
train_polynomial =np.poly1d(train_over_fitting)
test_polynomial =np.poly1d(test_over_fitting)

#outputs
model_final = model_polynomial(model)
y_train =  train_polynomial(Y_train)
y_test = test_polynomial(Y_test)


#show in graphıcs
plt.figure
plt.title("100 Data")
plt.plot(X_train,train_model,'go',model,model_final,'r-')
plt.plot(X_test,test_model,'bo')
plt.legend(('Training Data','Model','Testing Data'))

# MSE Loss Function
L_2 = 0.5*(np.sum(y_train-train_model)**2)
print("Training Error:",L_2)  # Training Error

# MSE Loss Function
L_2_test = 0.5*(np.sum(y_test-test_model)**2)
print("Testing Error:",L_2_test)  # Testing Error


# In[21]:


M = 10 #degree of polynomial

data = 1000
model = np.linspace(0,1,num=data)
train_data = np.linspace(0,1,num=data)
test_data = np.linspace(0,1,num=data)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, test_data, test_size = 0.20)

f1_model = np.sin(2*np.pi*model)
f1_train = np.sin(2*np.pi*X_train)
f1_test  = np.sin(2*np.pi*X_test)
#len(X_train) = 800
#len(X_test) = 200
gaussian_model = np.random.normal(0,0.05,data)
gaussian_train = np.random.normal(0,0.05,len(X_train))
gaussian_test = np.random.normal(0,0.05,len(X_test))

t_model = f1_model + gaussian_model
train_model = f1_train + gaussian_train
test_model = f1_test + gaussian_test

# fitting the 10th polynomial 
p = np.polyfit(model,f1_model,M)
train_over_fitting =np.polyfit(X_train, train_model,M)
test_over_fitting  =np.polyfit(X_test, test_model,M)

#create polynomial in numpy = np.poly1d
model_polynomial =np.poly1d(p)
train_polynomial =np.poly1d(train_over_fitting)
test_polynomial =np.poly1d(test_over_fitting)
#outputs
model_final = model_polynomial(model)
y_train =  train_polynomial(Y_train)
y_test = test_polynomial(Y_test)


#show in graphıcs
plt.figure
plt.title("100 Data")
plt.plot(X_train,train_model,'go',model,model_final,'r-')
plt.plot(X_test,test_model,'bo')
plt.legend(('Training Data','Model','Testing Data'))

# MSE Loss Function
L_2 = 0.5*(np.sum(y_train-train_model)**2)
print("Training Error:",L_2)  # Training Error

# MSE Loss Function
L_2_test = 0.5*(np.sum(y_test-test_model)**2)
print("Testing Error:",L_2_test)  # Testing Error


# In[ ]:




