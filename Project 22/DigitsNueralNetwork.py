import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import sys
sys.stdout.reconfigure(encoding='utf-8')


(xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()

xtrain = xtrain/255
xtest = xtest/255


xtrain = xtrain.reshape(len(xtrain), 28*28)
xtest = xtest.reshape(len(xtest), 28*28)

# print(f'xtrain: {xtrain.shape}\nytrain: {ytrain.shape}\nxtest: {xtest.shape}\nytest: {ytest.shape}\n')

model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(xtrain, ytrain, epochs=5)

predicitons = model.predict(xtest)
labelsP = [np.argmax(i) for i in predicitons]
cm = tf.math.confusion_matrix(labels=ytest,predictions=labelsP)

sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()