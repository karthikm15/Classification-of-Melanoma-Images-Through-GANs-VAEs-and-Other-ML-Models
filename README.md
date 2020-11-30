# Semi-Supervised-GANs-For-Melanoma-Detection

## Prerequisites: 

This liveProject is intended for intermediate Python programmers with at least some deep learning experience, preferably in image classification with convolutional
neural networks. Knowledge of PyTorch would be helpful, but is not required. No prior experience in generative modeling, including GANs, is assumed. 

### Tools: 

- Basics of PIL
- Basics of Matplotlib
- Intermediate Python
- Intermediate NumPy

### Techniques:

- Classification as a machine learning task
- Intermediate deep learning concepts such as convolutional neural networks

## Python Libraries:

- Torch 1.1.0: PyTorch, one of the most popular and fastest-growing deep learning frameworks
- Torchvision 0.3.0: PyTorch-based package for computer vision
- NumPy 1.15.4: a general-purpose mathematical package optimised for scientific computing
- Sklearn 0.20.1: for some basic data science utilities
- Matplotlib 3.0.2: for visualizing the data and the metrics
- Tensorflow 2.2.0: for building the semisupervised GAN framework

## Dataset:

The dataset can be found at https://lp-prod-resources.s3.amazonaws.com/other/MelanomaDetection.zip. It contains the melanocytic nevus image dataset separated into three folders: labeled, unlabeled and test.
The images are a pre-processed subset of a dataset available through the Dataverse project. All images are color 32x32 pixel JPG files. 
The naming convention for images in the labeled and test folders is as follows: each filename ends with either *_0.jpg or *_1.jpg, corresponding to a melanoma-negative 
or melanoma-positive image respectively. The number of files is as follows:

- _Labeled:_ 200 (evenly split between melanoma positive and negative)
- _Test:_ 600 (also evenly split between melanoma positive and negative)
- _Unlabeled:_ 7018 (class distribution unknown)

## Project:

Since there were only 200 labeled images to work with, data augmentation was performed on the training images (factoring their contrast, brightness,
sharpness, and color) to allow for larger batch sizes and more training images for the model to train on. Additionally, the data was resized to a 100x100
image so that the training time of the model is reduced.

### Supervised Image Classifier:

Using a Tensorflow backend, the supervised image classifier achieved an accuracy of ~75%. Here are the steps taken to implement the model:
- Build training and testing image data generators to feed images to the model in batches (batch size = 32).
- Define the model (2 convolutional layers and 4 max pooling layers).
- Train the model (using 10 epochs).
- Run the model through the testing images and find the accuracy of the model.
The model can be found in "Training a Supervised Image Classifier.ipynb".

## Semi-Supervised Image Classifier:

The semi-supervised classifier achieved an accuracy of ~75%. Here were the steps taken to implement the model.
- Train a supervised image classifier with the labeled images using the steps outlined above.
- Run predictions using the trained classifier on the unlabeled dataset.
- Take those labels provided to the unlabeled dataset and attach it to them. Re-run the model with the “labeled” unlabeled data and the labeled data.
- Find the accuracy by running it on the testing dataset.
The model used Tensorflow with a Keras backend to run. The code is stored in “Training a Semi-Supervised Image Classifier”.ipynb


## Variational Autoencoders:

The variational auto encoder trained on the MNIST dataset due to computational expenses.  Here were the steps taken to implement the model:
- Read in the MNIST dataset, define train and test datasets, and determine activation function.
- Create the encoder using a series of convolutional and dropout layers.
- Create the decoder using a series of dense, convolutional, and dropout layers.
- Make and compile the model with an Adam optimizer.
- Train the model and evaluate the images created by the decoder.
The model used Tensorflow with a Keras backend to run. The coder is stored in “Variational Autoencoder for MNIST Dataset (No Training Output)”.ipynb

### Semi-Supervised GAN:
Using a Tensorflow and Keras backend, the GAN achieved an accuracy of ~50%. The most probable reason for this low accuracy was the overarching strength
of the discriminator compared to the generator. Here are the steps taken to implement the model:
- Build training and testing image data generators using a class to feed images to the model in batches (batch size = 16).
- Build the generator network (three convolution layers and a hyperbolic tangent activation function).
- Build the discriminator network (3 convolutional layers with LeakyRELU and batch normalization).
- Build the unsupervised and supervised discriminator (softmax and a defined predict function respectively).
- Build the generative adversarial network combining both the generator and discriminator network.
- Train the model (using 500 iterations).
- Evaluate the model using the testing set.
The model can be found in "Semi_Supervised_GAN.ipynb".
