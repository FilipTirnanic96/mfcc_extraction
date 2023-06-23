import numpy as np
import scipy.io.wavfile as sci_wav # Open wav files
from matplotlib import pyplot as plt

from core_functions.mfcc_feature_extraction import extract_mfcc_feature
from core_functions.mfcc_utility_functions import stft, signal_power_to_db

import librosa
import librosa.display as libdisplay


def visualize_spectrogram(y_spec: np.array, parameters: dict, title: str = ""):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    libdisplay.specshow(y_spec, y_axis='linear', sr=parameters['fs'], cmap='autumn', x_axis='time', hop_length=parameters['window_step'])
    plt.show(block=False)


def visualize_mfcc(y_spec: np.array, parameters: dict, title: str = ""):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    libdisplay.specshow(y_spec, y_axis='frames', sr=parameters['fs'], x_axis='time', hop_length=parameters['window_step'])
    plt.colorbar()
    plt.ylabel('MFCC')
    plt.show(block=False)


def get_spectrogram_of_signal(signal: np.array, parameters: dict):
    """
    Get Spectrogram form librosa and algorithm implementation.

    :param signal: input signal
    :param parameters: Params used for Spectrogram extraction
    :return: Spectrogram from librosa and implementation
    """

    n_fft = parameters['n_fft']
    window_length = parameters['window_length']
    window_step = parameters['window_step']

    # Use librosa for spectrogram extraction
    signal_spec_lib = librosa.amplitude_to_db(np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=window_step,
                                                                  win_length=window_length)))

    frame_size = window_length / fs
    frame_step = window_step / fs
    # Use implemented function for spectrogram extraction
    power_spectrum_frames = np.abs(stft(y=signal, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step))**2
    signal_spec = signal_power_to_db(power_spectrum_frames)

    return signal_spec_lib, signal_spec.T


def get_mfcc_of_signal(signal: np.array, parameters: dict):
    """
    Get MFCC coefficient form librosa and algorithm implementation.

    :param signal: input signal
    :param parameters: Params used for MFCC extraction
    :return: MFCC from librosa and implementation
    """

    n_fft = parameters['n_fft']
    window_length = parameters['window_length']
    window_step = parameters['window_step']
    n_mels = 40
    n_mfcc = 13
    # Use librosa for MFCC extraction
    signal_mfcc_lib = librosa.feature.mfcc(sr=fs, y=signal, n_mfcc=n_mfcc, n_mels=n_mels, win_length=window_length,
                                           hop_length=window_step, lifter=22)[1:n_mfcc, :]

    frame_size = window_length / fs
    frame_step = window_step / fs
    # Use implemented function for MFCC extraction
    signal_mfcc = extract_mfcc_feature(y=signal, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step)

    return signal_mfcc_lib, signal_mfcc.T


if __name__ == '__main__':

    wav_audio_location = "../data/train/cat/cat_10.wav"
    # read sound data
    fs, y = sci_wav.read(wav_audio_location)
    # convert signal to double
    y = 1.0 * y
    t = np.linspace(0, y.shape[0] * fs, y.shape[0])

    # visualize signal
    plt.figure(figsize=(6, 4))
    plt.title("Cat sound audio wav")
    plt.plot(t, y)
    plt.show(block=False)

    # set parameters
    n_fft = 512
    params = {
        'fs': fs,
        'n_fft': n_fft,
        'window_length': n_fft,
        'window_step': int(n_fft/3),
    }

    # show spectrogram
    y_spec_lib, y_spec = get_spectrogram_of_signal(y, params)
    visualize_spectrogram(y_spec, params, "Implemented Spectrogram function")
    visualize_spectrogram(y_spec_lib, params, "Librosa Spectrogram function")

    # show MFCC
    y_mffc_lib, y_mfcc = get_mfcc_of_signal(y, params)
    visualize_mfcc(y_mfcc, params, "Implemented MFCC function")
    visualize_mfcc(y_mffc_lib, params, "Librosa MFCC function")

    plt.show()
