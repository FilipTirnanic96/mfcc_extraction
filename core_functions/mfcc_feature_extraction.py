import numpy as np
from core_functions.mfcc_utility_functions import mel_filter, discrete_cos_transformation, sin_liftering, stft, \
    signal_power_to_db


def extract_mfcc_feature(y: np.array, fs: float, n_fft: int = 512, frame_size: float = 0.025, frame_step: float = 0.01,
                         n_mels: int = 40, n_mfcc: int = 13):
    """
    Extracts  MFCCs from input signal y.

    :param y: Input signal
    :param fs: Sampling frequency
    :param n_fft: Num of points to calculate FFT
    :param frame_size: Size of each frame in seconds
    :param frame_step: Steps between frames in seconds
    :param n_mels: Number of Mel filters
    :param n_mfcc: Number of MFCC features
    :return: MFCC features of input signal
    """

    # mag calculate spectrogram of signal
    mag_spec_frames = np.abs(stft(y, fs, n_fft, frame_size, frame_step))

    # create power spectrogram
    pow_spec_frames = (mag_spec_frames**2) / mag_spec_frames.shape[1]

    # filter signal with Mel-filter
    mel_power_spec_frames, hz_freq = mel_filter(pow_spec_frames, 0, fs/2, n_mels, fs)

    # log of signal
    log_spec_frames = signal_power_to_db(mel_power_spec_frames)

    # perform DCT
    mfcc = discrete_cos_transformation(log_spec_frames)

    # liftering of mfcc
    mfcc = sin_liftering(mfcc)

    # take just lwo frequencies mfcc
    mfcc = mfcc[:, 1:n_mfcc]

    return mfcc