#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from PIL import Image
from matplotlib import pyplot as plt
import pickle


# In[151]:


x_train = [] #to hold our x values or (28x28) images
y_train = [] # to hold our upscaled (56x56) images 
files = glob.glob('data/train/*.jpg')  #path to the traiing jpgs.


# In[152]:


for f in files: #opening up all of our files to an array 
    img = Image.open(f).convert('L').resize((28, 28))
    x_train.append(np.array(img))
for f in files:
    img = Image.open(f).convert('L').resize((56, 56))
    y_train.append(np.array(img))


# In[153]:


x_train = np.array(x_train) #creating np arrays
y_train = np.array(y_train)


# In[154]:


x_train.shape #shape


# In[155]:


y_train.shape #shape


# In[140]:


x_train = x_train.astype('float32') / 255. #normalizing our values
y_train = y_train.astype('float32') / 255.


# In[158]:


x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) #reshaping  x and y train
x_train.shape
y_train = np.reshape(y_train, (len(y_train), 56, 56, 1))
y_train.shape


# In[193]:


mse = tf.keras.losses.MeanSquaredError() #setting our los function to mean squared error


# In[400]:


#the model 
from tensorflow.keras import layers
input_img = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(128, (4, 4), activation='relu', padding='same')(input_img)
x = layers.Conv2D(64, (4, 4), activation='relu', padding='same')(input_img)
x = layers.Conv2D(18, (4, 4), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)



x = layers.Conv2D(8, (4, 4), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (4,4), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(18, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((1, 1))(x)
x = layers.Conv2D(64, (4, 4), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(128, (4, 4), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)


autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


# In[401]:


autoencoder.summary() #summary of the model.


# In[402]:


autoencoder.fit(x_train, y_train,   #fitting the model to our x and y training data
                epochs=100,
                batch_size=128,
                shuffle=True)


# In[403]:


autoencoder.save('data/autoencoder_model.h5') #saving the model so we can load it below


# In[426]:


from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img
orig_image = Image.open('data/train/cat.50.jpg') #loading in an image
orig_image = orig_image.convert("L") #coverting it to b/w using PIL

model = load_model('data/autoencoder_model.h5')

pred_image = np.array(orig_image.copy().resize((28, 28)))
generated = model.predict(pred_image.reshape((1, 28, 28, 1)))
truth = orig_image.copy().resize((56, 56))
generated_image = np.reshape(generated, (56, 56, 1))
generated_image = array_to_img(generated_image)
generated_image = np.array(generated_image)
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.imshow(pred_image)
ax2.imshow(generated_image)
ax3.imshow(truth)
#plt.savefig('upscaled100.jpg')
plt.show()

