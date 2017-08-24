# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:01:31 2017

@author: Alla.Khramkova
"""

import keras
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
#from sklearn.decomposition import PCA
file = 'C:/Users/Alla.Khramkova/Documents/Data Science/Digit Recognizer/train.csv'
print ("Reading in and transforming data...")
df = pd.read_csv(file)
data = df.as_matrix().astype(np.float32)
np.random.shuffle(data)
X = data[:, 1:]
mu = X.mean(axis=0)
#X = X - mu # center the data
#pca = PCA()
#Z = pca.fit_transform(X)
y = data[:, 0]
yy = pd.get_dummies(y)
yyy = yy.as_matrix().astype(np.int)
#print ( yyy.shape)
#print (yyy[1,])
#print (X.shape)
#print (X[1,])
im = X[20, ]
im_sq = np.reshape(im, (28, 28))

# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()
#import tensorflow as tf
# Create the model: model

model = Sequential()
    
# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape = (784,)))
    
# Add the second hidden layer



model.add(Dense(50, activation='relu'))
# Add the third hidden layer
model.add(Dense(50, activation='relu'))
# Add the forth hidden layer
model.add(Dense(50, activation='relu'))
# Add the fifth layer
model.add(Dense(50, activation='relu'))
# Add the output layer
model.add(Dense(10, activation='softmax'))
    
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    
# Fit the model
hist = model.fit(X,yyy,validation_split=0.3)
model_json = model.to_json()
json_file = open("mnist_model.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("mnist_model_weights.h5")


"""
save_best_weights = "weights-pairs1.h5"

checkpoint = ModelCheckpoint('mnist_model', monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]
start = time.time()
model_info=merged_model.fit([x1, x2, x1, x2, x1, x2], y=y, batch_size=64, epochs=3, verbose=True,
validation_split=0.33, shuffle=True, callbacks=callbacks)
end = time.time()
print("Minutes elapsed: %f" % ((start - end) / 60.))

#####evaluting

#load weights into a new model

model_info.load_weights(save_best_weights)
predictions = model.predict([test_q1, test_q2, test_q1, test_q2,test_q1, test_q2], verbose = True)

file_test = 'C:/Users/Alla.Khramkova/Documents/Data Science/Digit Recognizer/test.csv'
print ("Reading in and transforming data...")
df_test = pd.read_csv(file_test)
data_test = df_test.as_matrix().astype(np.float32)
np.random.shuffle(data_test)
X_test = data_test[:, 1:]
mu_test = X_test.mean(axis=0)
#X = X - mu # center the data
#pca = PCA()
#Z = pca.fit_transform(X)
y_test = data_test[:, 0]
yy_test = pd.get_dummies(y_test)
yyy_test = yy_test.as_matrix().astype(np.int)
#print ( yyy.shape)
#print (yyy[1,])
#print (X.shape)
#print (X[1,])
im_test = X_test[20, ]
im_sq_test = np.reshape(im, (28, 28))

# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq_test, cmap='Greys', interpolation='nearest')
plt.show()
"""
"""
def save_model(model, options):
    json_string = model.to_json()
    open(options['file_arch'], 'w').write(json_string)
    model.save_weights(options['file_weight'])
def load_model(options):
    model = model_from_json(open(options['file_arch']).read())
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.load_weights(options['file_weight'])
    return model   
"""
