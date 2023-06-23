import os

import librosa
import numpy as np
import scipy.io.wavfile as sci_wav
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from core_functions.mfcc_feature_extraction import extract_mfcc_feature
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns

# Define data set root paths
fs = 16000  # 16kHz sampling rate
ROOT_DIR_CAT = '../data/train/cat/'
ROOT_DIR_CAT_TEST = '../data/test/cat/'

ROOT_DIR_DOG = '../data/train/dog/'
ROOT_DIR_DOG_TEST = '../data/test/dog/'


def read_wav_files(root_dir: str, wav_files: str):
    """
    Read audio data from provided paths.

    :param root_dir: Path to root directory
    :param wav_files: Name of .wav files to read
    :return: Audio data
    """

    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(root_dir + f)[1] for f in wav_files]


def read_train_test_data():
    """
    Read train and test data

    :return: Train and test data samples with labels
    """

    # list dir
    wav_files_cat = os.listdir(ROOT_DIR_CAT)
    wav_files_dog = os.listdir(ROOT_DIR_DOG)

    wav_files_cat_test = os.listdir(ROOT_DIR_CAT_TEST)
    wav_files_dog_test = os.listdir(ROOT_DIR_DOG_TEST)

    # read .wav files
    data_cats_train = read_wav_files(ROOT_DIR_CAT, wav_files_cat)
    data_dogs_train = read_wav_files(ROOT_DIR_DOG, wav_files_dog)

    data_cats_test = read_wav_files(ROOT_DIR_CAT_TEST, wav_files_cat_test)
    data_dogs_test = read_wav_files(ROOT_DIR_DOG_TEST, wav_files_dog_test)

    # make train and test data sets
    ds_train = data_dogs_train + data_cats_train
    labels_train = np.concatenate((np.ones(len(data_dogs_train)), np.zeros(len(data_cats_train))))

    ds_test = data_dogs_test + data_cats_test
    labels_test = np.concatenate((np.ones(len(data_dogs_test)), np.zeros(len(data_cats_test))))

    return ds_train, labels_train, ds_test, labels_test


def pad_data(data_set: list, fix_length: int):
    """
    Pad each sample from data set to fix length

    :param data_set: Input data
    :param fix_length: Fix length in number of samples to pad data
    :return: Data set with fix length
    """

    if not isinstance(data_set, list):
        return librosa.util.fix_length(data_set, size=fix_length, axis=0, mode='wrap')

    else:
        data_set_fix_length = np.zeros((len(data_set), fix_length))
        for i, data in enumerate(data_set):
            data_set_fix_length[i, :] = librosa.util.fix_length(data, size=fix_length, axis=0, mode='wrap')
    return data_set_fix_length


def mfcc_extraction(data_set_fix_length: list, fs: float, n_fft: int, frame_size: float, frame_step: float):
    """
    Extract MFCC features for each data sample.

    :param data_set_fix_length: Input data set
    :param fs: Sampling frequency
    :param n_fft: Num of Nfft points
    :param frame_size: Size of frame in sec
    :param frame_step: Frame step in sec
    :return: MFCC features for each data sample
    """

    flag = True
    for i, data in enumerate(data_set_fix_length):
        mfcc = extract_mfcc_feature(y=data, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step)
        if flag:
            mfcc_features = np.zeros((len(data_set_fix_length), mfcc.shape[0], mfcc.shape[1]))
            flag = False
        mfcc_features[i, :, :] = mfcc
    return mfcc_features


def preprocess_raw_data(data: list, fix_length: int, mfcc_parameters: dict):
    """
    Preprocess audio data. Returns MFCC features of each audio sample.

    :param data: List of audio data
    :param fix_length: Fix length to pad each audio sample
    :param mfcc_parameters: Parameters for MFCC extraction
    :return: MFCC data for each audio sample
    """

    # pad data to fix length
    data_set_fix_length = pad_data(data, fix_length)
    # extract MFCC for each sample
    mfcc_features = mfcc_extraction(data_set_fix_length, mfcc_parameters['fs'], mfcc_parameters['n_fft'],
                                    mfcc_parameters['frame_size'], mfcc_parameters['frame_step'])

    return mfcc_features


def show_confusion_matrix(title: str, confusion_matrix, class_names: list):
    """
    Generates plot of confusion matrix

    :param title: Plot title
    :param confusion_matrix: Confusion matrix
    :param class_names: List of class names
    """

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

    plt.figure(figsize=(7, 5))
    plt.title(title)

    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show(block=False)


if __name__ == '__main__':

    # Load data
    data_set_train, y_train, data_set_test, y_test = read_train_test_data()

    # Set parameters
    n_fft = 512
    mfcc_parameters = {
        'fs': fs,
        'n_fft': n_fft,
        'frame_size': n_fft / fs,
        'frame_step': int(n_fft / 3) / fs,
    }

    # Pick the fixed length in seconds\
    fix_len_s = 13.5  # 13.5 s
    fix_len = int(fix_len_s * fs)

    # Extract MFCC features
    X_train = preprocess_raw_data(data_set_train, fix_len, mfcc_parameters)
    X_train = np.array([feature.ravel() for feature in X_train])
    X_test = preprocess_raw_data(data_set_test, fix_len, mfcc_parameters)
    X_test = np.array([feature.ravel() for feature in X_test])

    # Use PCA
    n_components = 30
    pca = PCA(n_components=n_components)

    X_train = pca.fit(X_train).transform(X_train)
    X_test = pca.transform(X_test)

    # Train logistic regression
    logreg = LogisticRegression(C=1e5, solver='lbfgs')

    y_train_pred = logreg.fit(X_train, y_train).predict(X_train)
    y_test_pred = logreg.predict(X_test)

    # Print reports
    print('Train classification report:')
    print(classification_report(y_train, y_train_pred))
    print('Test classification report:')
    print(classification_report(y_test, y_test_pred))

    # Show confusion matrix
    show_confusion_matrix("Train data", confusion_matrix(y_train, y_train_pred), ['cats', 'dogs'])
    show_confusion_matrix("Test data", confusion_matrix(y_test, y_test_pred), ['cats', 'dogs'])
    plt.show()
