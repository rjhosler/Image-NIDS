# This is the code used for our paper: Unsupervised Deep Learning for an Image Based Network Intrusion Detection System.
Our implementations for the Wasserstein Autoencoder, Wasserstein BiGAN, and creating images can be found in this repository.

## environment.yml
This code is intended to run in an anaconda virtual environment. To install all required packages, run the following command:

`conda env create -f environment.yml`

## create_bot.py
This file contains our implementation of converting feature vectors from the [Bot-IoT](https://research.unsw.edu.au/projects/bot-iot-dataset) dataset into images. 
A directory of images can be created by running this command:

`python create_images.py`

## wae.py
This file will train a Wasserstein Autoencoder model and save checkpoints to the Models directory by running this command:

`python wae.py`

## bigan.py
This file will train a Wasserstein BiGAN model and save checkpoints to the Models directory by running this command:

`python bigan.py`

## Channel-Vectors
This directory contains the channel vectors that are used for converting feature vectors into images.
