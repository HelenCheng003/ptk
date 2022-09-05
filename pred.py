import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import wave
import pylab
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization

FILE_LOC = "" #Set your file loc

PRED_INPUT_DIR = FILE_LOC + "\\ptk\\testwaves"
PRED_OUTPUT_DIR = FILE_LOC + "\\ptk\\testimg"
#wav_to_specto(PRED_INPUT_DIR, PRED_OUTPUT_DIR, "pred")

load_model = tf.keras.models.load_model(FILE_LOC + "\\ptk\\model.710-0.00.h5")

def predict_ptk_count(predict_files_path, model, ptk_count):
    testfile_list = os.listdir(predict_files_path)
    prediction_list = []
    for i in range(len(testfile_list)):
        img_loc = os.path.join(predict_files_path,testfile_list[i])
        img = tf.keras.preprocessing.image.load_img(img_loc, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        prediction = model.predict(img_array)
        prediction = np.where(prediction > 0.8, 1, 0)
        if np.array_equal([[1, 0, 0]], prediction):
            prediction_list.append("k")
        elif np.array_equal([[0, 1, 0]], prediction):
                prediction_list.append("p")
        elif np.array_equal([[0, 0, 1]], prediction):
                prediction_list.append("t")
    for i in range(len(prediction_list)-3):
        if (prediction_list[i] == "p") and (prediction_list[i+1] == "t") and (prediction_list[i+2] == "k"):
            ptk_count = ptk_count + 1

ptk_count = 0        
predict_ptk_count(os.path.join(PRED_OUTPUT_DIR, 'audio-images'), load_model, ptk_count)

print("ptk counts = " + str(ptk_count))




