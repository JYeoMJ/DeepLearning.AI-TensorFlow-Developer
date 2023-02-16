# Introduction to TensorFlow for AI, Machine Learning and Deep Learning
# Course 1 of 4 in the DeepLearning.AI TensorFlow Developer Specialization

## Introduction to Image Classification with Neural Networks (Fashion MNIST)

import tensorflow as tf
print(tf.__version__)

# Terminate training at desired accuracy/loss level

class myCallback(tf.keras.callbacks.Callback): # use to define methods to set where callback will be executed
  def on_epoch_end(self, epoch, logs={}):
    
    '''
    Halts the training after reaching 60 percent accuracy

    Args:
      epoch (integer) - index of epoch (required but unused in the function definition below)
      logs (dict) - metric results from the training epoch
    '''

    if(logs.get('accuracy') >= 0.6): # <--- Alternative: if (logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback() # Instantiate Class!

# Loading Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

print(np.shape(training_images[0])) # MNIST data in form of 28x28 grayscale images

# Visualizing Input Data
import numpy as np
import matplotlib.pyplot as plt

index = 0 # Select index between 0 to 5999
np.set_printoptions(linewidth=320)
print(f'LABEL: {training_labels[index]}') # ground truth
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

plt.imshow(training_images[index])

# Normalize pixel values, scale all values to between 0 and 1
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), # Hidden layer
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) # Output layer, probability scores

# Model training and fitting
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks]) # Opt: exclude callbacks argumens

# Evaluate the model on unseen data
model.evaluate(test_images, test_labels)

# Predictions
classifications = model.predict(test_images)
p_score = np.max(classifications[0]); pred = np.argmax(classifications[0])
print(f'probability score: {p_score}')
print(f'class with highest probability: {pred}')
print(classifications[0])

# Hyperparameter tuning options

"""
> Increasing no. of nodes in hidden layer
> Increasing layers
> Increasing/Decreasing no. of epochs (caution on model overfitting)

"""

## Additional Notes ------------------------------------------

##  Softmax function principles (generating probability scores)

# Declare sample inputs and convert to a tensor
inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f'input to softmax function: {inputs.numpy()}')

# Feed the inputs to a softmax activation function
outputs = tf.keras.activations.softmax(inputs)
print(f'output of softmax function: {outputs.numpy()}')

# Get the sum of all values after the softmax
sum = tf.reduce_sum(outputs)
print(f'sum of outputs: {sum}')

# Get the index with highest value
prediction = np.argmax(outputs)
print(f'class with highest probability: {prediction}')

## -----------------------------------------------------

##########
# Week 3 #
##########

## Enhancing Image Classification with Convolutional Neural Networks (CNNs)

import tensorflow as tf

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels) , (test_images, test_labels) = fmnist.load_data()

# Normalize pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0

# First consider the Shallow Neural Network

model = tf.keras.models.Sequential({
  tf.keras.layers.Flatten()
  tf.keras.layers.Dense(128, activation = tf.nn.relu)
  tf.keras.layers.Dense(10, activation = tf.nn.softmax)
  })

# Setup training parameters

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Train the model

print(f'\nMODEL TRAINING:' )
model.fit(training_images, training_labels, epochs = 5)

# Evaluate on the test set

print(f'\nMODEL EVALUATION:' )
model.evaluate(test_images, test_labels)

# Next consider the CNN model

"""
Using Convolutional Neural Networks for feature extraction, 
improving classification accuracy. Applying a convolutional filter. 
Combine with pooling operations to reduce the dimensionality of the problem.

"""
model = tf.keras.models.Sequential([

  # Add convolutions and max pooling
  tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  # Add the same layers as before
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation = 'relu'),
  tf.keras.layers.Dense(10, acitvation = 'softmax')
  ])

# Print model summary
model.summary() # Note to pay attention to the output shape column!

# Use same settings
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Train the model

print(f'\nMODEL TRAINING:' )
model.fit(training_images, training_labels, epochs = 5)

# Evaluate on the test set

print(f'\nMODEL EVALUATION:' )
model.evaluate(test_images, test_labels)

##################################################
# Visualizing effects of Convolution and Pooling #
##################################################

print(test_labels[:100]) # prints first 100 labels in test
np.where(test_labels[:100] == 9) # identifying position of labels corresponding to shoes

import matplotlib.pyplot as plt
from tensorflow.keras import models

f, axarr = plt.subplots(3,4)

FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28

# Recall we have 32 feature maps/convolutional filters, each to extract specific feature
# During training, CNN learns the optimal values for filter weights to identify the most relevant features
CONVOLUTION_NUMBER = 1 

# Create list of each layers output
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

# Observe the results for each layer
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)

# --------------------------------------------------------------

## Exploring Convolutions 

import scipy # Load ascent image
ascent_image = scipy.datasets.ascent()

import matplotlib.pyplot as plt

# Visualize the image
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(ascent_image)
plt.show()

import numpy as np

# Copy image into numpy array
image_transformed = np.copy(ascent_image)

# Get the dimensions of the image
size_x = image_transformed.shape[0]
size_y = image_transformed.shape[1]

# Experiment with different values and see the effect
filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]

# A couple more filters to try for fun!
# filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]] # emphasizes vertical lines
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] # horizontal lines

# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1

# Iterate over the image
for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      convolution = 0.0
      convolution = convolution + (ascent_image[x-1, y-1] * filter[0][0])
      convolution = convolution + (ascent_image[x-1, y] * filter[0][1])  
      convolution = convolution + (ascent_image[x-1, y+1] * filter[0][2])     
      convolution = convolution + (ascent_image[x, y-1] * filter[1][0])    
      convolution = convolution + (ascent_image[x, y] * filter[1][1])    
      convolution = convolution + (ascent_image[x, y+1] * filter[1][2])    
      convolution = convolution + (ascent_image[x+1, y-1] * filter[2][0])    
      convolution = convolution + (ascent_image[x+1, y] * filter[2][1])    
      convolution = convolution + (ascent_image[x+1, y+1] * filter[2][2])    
      
      # Multiply by weight
      convolution = convolution * weight   
      
      # Check the boundaries of the pixel values
      if(convolution<0):
        convolution=0
      if(convolution>255):
        convolution=255

      # Load into the transformed image
      image_transformed[x, y] = convolution

# Plot the image. Note the size of the axes -- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(image_transformed)
plt.show()   

# Assign dimensions half the size of the original image
new_x = int(size_x/2)
new_y = int(size_y/2)

# Create blank image with reduced dimensions
newImage = np.zeros((new_x, new_y))

# Iterate over the image
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    
    # Store all the pixel values in the (2,2) pool
    pixels = []
    pixels.append(image_transformed[x, y])
    pixels.append(image_transformed[x+1, y])
    pixels.append(image_transformed[x, y+1])
    pixels.append(image_transformed[x+1, y+1])

    # Get only the largest value and assign to the reduced image
    newImage[int(x/2),int(y/2)] = max(pixels)

# Plot the image. Note the size of the axes -- it is now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.show()      

## -----------------------------------------------------

### UNDERSTANDING `ImageDataGenerator`
# automatically load and label files based on their subdirectories

train_datagen = ImageDataGenerator(rescale=1./255)

# Point at main directory containing sub-directory, that contains your images

train_generator = train_datagen.flow_from_directory(
  train_dir, # point at correct dir
  target_size = (300,300), # resize data to be uniformly sized
  batch_size = 128,
  class_mode = 'binary') # for binary classifiers

validation_generator = train_datagen.flow_from_directory(
  validation_dir,
  target_size = (300,300),
  batch_size = 32,
  class_mode = 'binary')


