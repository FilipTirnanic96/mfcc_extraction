import numpy as np
from core_functions.mfcc_utility_functions import mel_filter, discrete_cos_transformation, sin_liftering, stft, \
    signal_power_to_db


def extract_mfcc_feature(y: np.array, fs: float, n_fft: int = 512, frame_size: float = 0.025, frame_step: float = 0.01):
    """
    Extracts  MFCCs from input signal y.

    :param y: Input signal
    :param fs: Sampling frequency
    :param n_fft: Num of points to calculate FFT
    :param frame_size: Size of each frame in seconds
    :param frame_step: Steps between frames in seconds
    :return: MFCC features of input signal
    """

    # mag calculate spectrogram of signal
    mag_spec_frames = np.abs(stft(y, fs, n_fft, frame_size, frame_step))

    # create power spectrogram
    pow_frames = (mag_spec_frames**2) / mag_spec_frames.shape[1]

    # filter signal with Mel-filter
    frames_envelop, hz_freq = mel_filter(pow_frames, 0, fs/2, 40, fs)

    # log of signal
    log_pow_frames = signal_power_to_db(frames_envelop)

    # perform DCT
    mfcc = discrete_cos_transformation(log_pow_frames)

    # liftering of mfcc
    mfcc = sin_liftering(mfcc)

    # take just lwo frequencies mfcc
    mfcc = mfcc[:, 1:13]

    return mfcc