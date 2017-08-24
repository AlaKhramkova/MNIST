# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:58:36 2017

@author: Alla.Khramkova
"""
import keras
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from keras.applications.resnet50 import decode_predictions
file_test = 'C:/Users/Alla.Khramkova/Documents/Data Science/Digit Recognizer/test.csv'
print ("Reading in and transforming data...")
df_test = pd.read_csv(file_test)
data_test = df_test.as_matrix().astype(np.float32)
np.random.shuffle(data_test)
X_test = data_test[:, 0:]
mu_test = X_test.mean(axis=0)
#X = X - mu # center the data
#pca = PCA()
#Z = pca.fit_transform(X)

#print ( yyy.shape)
#print (yyy[1,])
print (X_test.shape)
#print (X[1,])
im_test = X_test[20, ]
im_sq_test = np.reshape(im_test, (28, 28))
# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq_test, cmap='Greys', interpolation='nearest')
plt.show()
model = load_model('mnist_model.h5')                                                                                
model.load_weights("mnist_model_weights.h5")
preds = model.predict(X_test)
print("Predicted:",np.argmax(preds[20,], axis=0))
