# Vocal_MNIST_predictor
This project objective is to predict voice content by converting it from an audio signal to an image and run it through a Convolution Neural Network (CNN).

# Citation:
Raw dataset is based on Free Spoken Digit Data (FSDD) dataset by: Zohar Jackson, César Souza, Jason Flaks, Yuxin Pan, Hereman Nicolas, & Adhish Thite. (2018).
Jakobovski/free-spoken-digit-dataset: v1.0.8 (v1.0.8). Zenodo. https://doi.org/10.5281/zenodo.1342401
audio file structure and naming: .wav file named 8_dror_0.wav where 8 is the files class / label, dror is authors name and 0 is a serial number of file.

Main process: take audio .wav file --> convert it from continuous signal to a descrete frequencies distribution using Fast Furier Transform --> represnt audio in a spectrogram image of frequencies over time --> convert spectrogram images to numeric representation and cranch it in a CNN training procedure to get a digit written identification of its FSDD audio file.
![audio_images1](https://user-images.githubusercontent.com/88071463/137535726-09c43ee7-60ea-44bd-9281-c7b47273014e.jpg)

Side process: run_your_audio is doing the same conversion from an audio to an image but using CNN as a predicition tool.

# License:
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)

# Structure:
main.py is the main code for processing FSDD dataset and CNN training.
data is a folder with an audio files zipped dataset (3000 files).
audio_images is a folder with created images in .jpg format and 'audio_image0.jpg' naming.
utils is a folder with few sub-folders:
1. my_model: trained CNN model params, weights etc.
2. WAV2IMG.py: functions file with .wav files read, display and convert to spectrogram image.
3. WAV_DIR_2_IMG_DIR.py: function file for audio files reading and handling and data preprations for CNN run.
4. CNN_model.py: functions for defining CNN model strucutre and its hyperparams, and for train-test spliting dataset.

# Requirements:
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import warnings
from matplotlib import pyplot as plt
from matplotlib import cm
from librosa import load,display
import scipy
from sklearn.preprocessing import MinMaxScaler

Enjoy
