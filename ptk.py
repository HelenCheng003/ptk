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
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization



# Set paths to input and output data
FILE_LOC = "" #Set your file loc
INPUT_DIR = FILE_LOC + "\\ptk\waves"
OUTPUT_DIR = FILE_LOC + "\\ptk\waves_img"




def split(save_path, wav_file_path, slen, sthresh):
    sound = AudioSegment.from_wav(wav_file_path)

    audio_chunks = split_on_silence(sound, min_silence_len=slen, silence_thresh=sthresh)

    for i, chunk in enumerate(audio_chunks):
       output_file = save_path + "\\"+ str(i).zfill(2) + ".wav".format(i)
       print("Exporting file", output_file)
       chunk.export(output_file, format="wav")

def stretch(data, rate=1):
    input_length = 1600
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def shfit(data):
    data = np.roll(data, 160)
    return data

def data_augment(input_path):
    file_list = os.listdir(input_path)
    stretch_const_list = [0.8, 1.2]
    for k in stretch_const_list:
        for j in range(len(stretch_const_list)):
            for i in range(len(file_list)):
                data = load_audio_file(os.path.join(input_path, file_list[i]))
                new_data = stretch(data, k)
                sf.write(os.path.join(INPUT_DIR, file_list[i][0:2]) + "_" + str(j) + ".wav", new_data, 1600)
    for i in range(len(file_list)):
        data = load_audio_file(os.path.join(input_path, file_list[i]))
        new_data = shfit(data)
        sf.write(os.path.join(INPUT_DIR, file_list[i][0:2]) + "_" + str(len(stretch_const_list)) + ".wav", new_data, 1600)
        
                
# data_augment(INPUT_DIR)



# Utility function to get sound and frame rate info
def get_wav_info(wav_file):
    wav = wave.open(wav_file, "r")
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, "int16")
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def wav_to_specto(input_dir, output_dir, mode):
    if not os.path.exists(os.path.join(output_dir, "audio-images")):
        os.mkdir(os.path.join(output_dir, "audio-images"))
        for filename in os.listdir(input_dir):
            if "wav" in filename:
                file_path = os.path.join(input_dir, filename)
                file_stem = Path(file_path).stem
                if mode == "train":
                    target_dir = str(int(filename[0]+filename[1]) % 3)
                elif mode == "pred":
                    target_dir = ""
                dist_dir = os.path.join(os.path.join(output_dir, "audio-images"), target_dir)
                file_dist_path = os.path.join(dist_dir, file_stem)
            if not os.path.exists(file_dist_path + ".png"):
                if not os.path.exists(dist_dir):
                    os.mkdir(dist_dir)
                file_stem = Path(file_path).stem
                sound_info, frame_rate = get_wav_info(file_path)
                pylab.specgram(sound_info, Fs=frame_rate)
                pylab.savefig(f"{file_dist_path}.png")
                pylab.close()
                
# wav_to_specto(INPUT_DIR, OUTPUT_DIR, "train")


# Declare constants
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 3

# Make a dataset containing the training spectrograms
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=os.path.join(OUTPUT_DIR, "audio-images"),
                                             shuffle=True,
                                             color_mode="rgb",
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="training",
                                             seed=0)

# Make a dataset containing the validation spectrogram
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=os.path.join(OUTPUT_DIR, "audio-images"),
                                             shuffle=True,
                                             color_mode="rgb",
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="validation",
                                             seed=0)



# Function to prepare our datasets for modelling
def prepare(ds):
    rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])
    ds = ds.map(lambda x, y: (rescale(x, training=True), y))
    return ds

train_dataset = prepare(train_dataset)
valid_dataset = prepare(valid_dataset)


# Create CNN model
model = tf.keras.models.Sequential()
#model.add(tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)))

model.add(Conv2D(32, 3, strides=2, padding="same", activation="relu", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(BatchNormalization())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(N_CLASSES, activation="softmax"))


#Compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)
model.summary()

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=OUTPUT_DIR + "\model\model.{epoch:02d}-{val_loss:.2f}.h5" ,save_best_only=True)
]


history = model.fit(train_dataset, epochs=1000, validation_data=valid_dataset, callbacks=my_callbacks)


final_loss, final_acc = model.evaluate(valid_dataset, verbose=0)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))