#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install sklearn


# In[3]:


pip install opencv-python


# In[4]:


pip install numpy


# In[5]:


pip install matplotlib


# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[36]:


import os
path=os.listdir('Downloads/braintumords')
classes={'no_tumor':0,'pituitary_tumor':1}


# In[37]:


import cv2
X = []
Y = []
for cls in classes:
    pth = 'Downloads/braintumords/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])


# In[38]:


X = np.array(X)
Y = np.array(Y)


# In[39]:


np.unique(Y)


# In[40]:


pd.Series(Y).value_counts()


# In[41]:


X.shape


# In[42]:


plt.imshow(X[0], cmap='gray')


# In[43]:


X_updated = X.reshape(len(X), -1)
X_updated.shape


# In[44]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,test_size=.20)


# In[45]:


xtrain.shape, xtest.shape


# In[46]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# In[47]:


from sklearn.decomposition import PCA


# In[48]:


print(xtrain.shape, xtest.shape)
pca = PCA(.98)
pca_train = xtrain
pca_test = xtest


# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[50]:


import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(pca_train, ytrain)


# In[51]:


sv = SVC()
sv.fit(pca_train, ytrain)


# In[52]:


print("Training Score:", lg.score(pca_train, ytrain))
print("Testing Score:", lg.score(pca_test, ytest))


# In[53]:


print("Training Score:", sv.score(pca_train, ytrain))
print("Testing Score:", sv.score(pca_test, ytest))


# In[54]:


pred1 = lg.predict(pca_test)
np.where(ytest!=pred1)


# In[55]:


pred = sv.predict(pca_test)
np.where(ytest!=pred)


# In[56]:


pred[36]


# In[57]:


ytest[36]


# In[58]:


accuracy_score(ytest,pred)*100


# In[59]:


accuracy_score(ytest,pred1)*100


# In[60]:


dec = {0:'No Tumor', 1:'Positive Tumor'}


# In[61]:


plt.figure(figsize=(12,8))
p = os.listdir('Downloads/braintumords/Testing/')
c=1
for i in os.listdir('Downloads/braintumords/Testing/no_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('Downloads/braintumords/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[62]:


plt.figure(figsize=(12,8))
p = os.listdir('Downloads/braintumords/Testing/')
c=1
for i in os.listdir('Downloads/braintumords/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('Downloads/braintumords/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[63]:


plt.figure(figsize=(12,8))
p = os.listdir('Downloads/braintumords/Testing/')
c=1
for i in os.listdir('Downloads/braintumords/Testing/no_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('Downloads/braintumords/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = lg.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[64]:


plt.figure(figsize=(12,8))
p = os.listdir('Downloads/braintumords/Testing/')
c=1
for i in os.listdir('Downloads/braintumords/Testing/pituitary_tumor/')[:25]:
    plt.subplot(5,5,c)
    
    img = cv2.imread('Downloads/braintumords/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = lg.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[65]:


plt.figure(figsize=(12,8))
p = os.listdir('Downloads/braintumords/Testing/')
c=1
for i in os.listdir('Downloads/braintumords/Testing/proj/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('Downloads/braintumords/Testing/proj/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[66]:


from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(ytest, pred)

print("Confusion Matrix:")
print(confusion_mat)


# In[68]:


from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(ytest, pred1)

print("Confusion Matrix:")
print(confusion_mat)


# In[ ]:


plt.figure(figsize=(12,8))
p = os.listdir('Downloads/braintumords/Testing/')
c=1
for i in os.listdir('Downloads/braintumords/Testing/proj/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('Downloads/braintumords/Testing/proj/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1

