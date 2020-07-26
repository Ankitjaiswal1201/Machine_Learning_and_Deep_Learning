
import keras

# ## Training a convnet from scratch on a small dataset
# 
# Having to train an image classification model using only very little data is a common situation, which you likely encounter yourself in 
# practice if you ever do computer vision in a professional context.
# 
# Having "few" samples can mean anywhere from a few hundreds to a few tens of thousands of images. As a practical example, we will focus on 
# classifying images as "dogs" or "cats", in a dataset containing 4000 pictures of cats and dogs (2000 cats, 2000 dogs). We will use 2000 
# pictures for training, 1000 for validation, and finally 1000 for testing.
# 
# In this section, we will review one basic strategy to tackle this problem: training a new model from scratch on what little data we have. We 
# will start by naively training a small convnet on our 2000 training samples, without any regularization, to set a baseline for what can be 
# achieved. This will get us to a classification accuracy of 71%. At that point, our main issue will be overfitting. Then we will introduce 
# *data augmentation*, a powerful technique for mitigating overfitting in computer vision. By leveraging data augmentation, we will improve 
# our network to reach an accuracy of 82%.
# 
# In the next section, we will review two more essential techniques for applying deep learning to small datasets: *doing feature extraction 
# with a pre-trained network* (this will get us to an accuracy of 90% to 93%), and *fine-tuning a pre-trained network* (this will get us to 
# our final accuracy of 95%). Together, these three strategies -- training a small model from scratch, doing feature extracting using a 
# pre-trained model, and fine-tuning a pre-trained model -- will constitute your future toolbox for tackling the problem of doing computer 
# vision with small datasets.


# ## Downloading the data
# 
# The cats vs. dogs dataset that we will use isn't packaged with Keras. It was made available by Kaggle.com as part of a computer vision 
# competition in late 2013, back when convnets weren't quite mainstream. You can download the original dataset at: 
# `https://www.kaggle.com/c/dogs-vs-cats/data` (you will need to create a Kaggle account if you don't already have one -- don't worry, the 
# process is painless).
# 
# The pictures are medium-resolution color JPEGs. They look like this:
# 
# ![cats_vs_dogs_samples](https://s3.amazonaws.com/book.keras.io/img/ch5/cats_vs_dogs_samples.jpg)

# Unsurprisingly, the cats vs. dogs Kaggle competition in 2013 was won by entrants who used convnets. The best entries could achieve up to 
# 95% accuracy. In our own example, we will get fairly close to this accuracy (in the next section), even though we will be training our 
# models on less than 10% of the data that was available to the competitors.
# This original dataset contains 25,000 images of dogs and cats (12,500 from each class) and is 543MB large (compressed). After downloading 
# and uncompressing it, we will create a new dataset containing three subsets: a training set with 1000 samples of each class, a validation 
# set with 500 samples of each class, and finally a test set with 500 samples of each class.
# 
# Here are a few lines of code to do this:

# In[2]:


import os, shutil


# In[ ]:


# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = 'D:\Ankit_Jaiswal\Extras\ComputerVision-Projects\kaggle_original_data'

# The directory where we will
# store our smaller dataset
base_dir = 'D:\Ankit_Jaiswal\Extras\ComputerVision-Projects\cats_and_dogs_small'
os.mkdir(base_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# Copy first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


# As a sanity check, let's count how many pictures we have in each training split (train/validation/test):

print('total training cat images:', len(os.listdir(train_cats_dir)))

print('total training dog images:', len(os.listdir(train_dogs_dir)))

print('total validation cat images:', len(os.listdir(validation_cats_dir)))

print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))


print('total test dog images:', len(os.listdir(test_dogs_dir)))


# 
# So we have indeed 2000 training images, and then 1000 validation images and 1000 test images. In each split, there is the same number of 
# samples from each class: this is a balanced binary classification problem, which means that classification accuracy will be an appropriate 
# measure of success.

# 
# Since we are attacking a binary classification problem, we are ending the network with a single unit (a `Dense` layer of size 1) and a 
# `sigmoid` activation. This unit will encode the probability that the network is looking at one class or the other.



from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# Let's take a look at how the dimensions of the feature maps change with every successive layer:

model.summary()

# For our compilation step, we'll go with the `RMSprop` optimizer as usual. Since we ended our network with a single sigmoid unit, we will 
# use binary crossentropy as our loss (as a reminder, check out the table in Chapter 4, section 5 for a cheatsheet on what loss function to 
# use in various situations).


from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# ## Data preprocessing
# 
# As you already know by now, data should be formatted into appropriately pre-processed floating point tensors before being fed into our 
# network. Currently, our data sits on a drive as JPEG files, so the steps for getting it into our network are roughly:
# 
# * Read the picture files.
# * Decode the JPEG content to RBG grids of pixels.
# * Convert these into floating point tensors.
# * Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know, neural networks prefer to deal with small input values).
# 
# It may seem a bit daunting, but thankfully Keras has utilities to take care of these steps automatically. Keras has a module with image 
# processing helper tools, located at `keras.preprocessing.image`. In particular, it contains the class `ImageDataGenerator` which allows to 
# quickly set up Python generators that can automatically turn image files on disk into batches of pre-processed tensors. This is what we 
# will use here.



from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


# Let's take a look at the output of one of these generators: it yields batches of 150x150 RGB images (shape `(20, 150, 150, 3)`) and binary 
# labels (shape `(20,)`). 20 is the number of samples in each batch (the batch size). Note that the generator yields these batches 
# indefinitely: it just loops endlessly over the images present in the target folder. For this reason, we need to `break` the iteration loop 
# at some point.



for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


# Let's fit our model to the data using the generator. We do it using the `fit_generator` method, the equivalent of `fit` for data generators 
# like ours. It expects as first argument a Python generator that will yield batches of inputs and targets indefinitely, like ours does. 
# Because the data is being generated endlessly, the generator needs to know example how many samples to draw from the generator before 
# declaring an epoch over. This is the role of the `steps_per_epoch` argument: after having drawn `steps_per_epoch` batches from the 
# generator, i.e. after having run for `steps_per_epoch` gradient descent steps, the fitting process will go to the next epoch. In our case, 
# batches are 20-sample large, so it will take 100 batches until we see our target of 2000 samples.
# 
# When using `fit_generator`, one may pass a `validation_data` argument, much like with the `fit` method. Importantly, this argument is 
# allowed to be a data generator itself, but it could be a tuple of Numpy arrays as well. If you pass a generator as `validation_data`, then 
# this generator is expected to yield batches of validation data endlessly, and thus you should also specify the `validation_steps` argument, 
# which tells the process how many batches to draw from the validation generator for evaluation.


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


# It is good practice to always save your models after training:


model.save('cats_and_dogs_small_1.h5')


# Let's plot the loss and accuracy of the model over the training and validation data during training:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# These plots are characteristic of overfitting. Our training accuracy increases linearly over time, until it reaches nearly 100%, while our 
# validation accuracy stalls at 70-72%. Our validation loss reaches its minimum after only five epochs then stalls, while the training loss 
# keeps decreasing linearly until it reaches nearly 0.
# 
# Because we only have relatively few training samples (2000), overfitting is going to be our number one concern. You already know about a 
# number of techniques that can help mitigate overfitting, such as dropout and weight decay (L2 regularization). We are now going to 
# introduce a new one, specific to computer vision, and used almost universally when processing images with deep learning models: *data 
# augmentation*.

# ## Using data augmentation
# 
# Overfitting is caused by having too few samples to learn from, rendering us unable to train a model able to generalize to new data. 
# Given infinite data, our model would be exposed to every possible aspect of the data distribution at hand: we would never overfit. Data 
# augmentation takes the approach of generating more training data from existing training samples, by "augmenting" the samples via a number 
# of random transformations that yield believable-looking images. The goal is that at training time, our model would never see the exact same 
# picture twice. This helps the model get exposed to more aspects of the data and generalize better.
# 
# In Keras, this can be done by configuring a number of random transformations to be performed on the images read by our `ImageDataGenerator` 
# instance. Let's get started with an example:



datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# This is module with image preprocessing utilities
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# We pick one image to "augment"
img_path = fnames[3]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()


# If we train a new network using this data augmentation configuration, our network will never see twice the same input. However, the inputs 
# that it sees are still heavily intercorrelated, since they come from a small number of original images -- we cannot produce new information, 
# we can only remix existing information. As such, this might not be quite enough to completely get rid of overfitting. To further fight 
# overfitting, we will also add a Dropout layer to our model, right before the densely-connected classifier:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# Let's train our network using data augmentation and dropout:

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

# Let's save our model -- we will be using it in the section on convnet visualization.

model.save('cats_and_dogs_small_2.h5')

# Let's plot our results again:

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Thanks to data augmentation and dropout, we are no longer overfitting: the training curves are rather closely tracking the validation 
# curves. We are now able to reach an accuracy of 82%, a 15% relative improvement over the non-regularized model.
