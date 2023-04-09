import numpy as np
from matplotlib import pyplot as plt

from core_functions.mffc_utility_functions import preemphasis_filter, framing, hamming, mel_filter, \
    discrete_cos_transformation, sin_liftering, stft


def extract_mffc_feature(y: np.array, fs: float, n_fft: int = 512, frame_size: float = 0.025, frame_step: float = 0.01):
    # mag calculate spectrogram of signal
    mag_spec_frames = np.abs(stft(y, fs, n_fft, frame_size, frame_step))

    # create power spectrogram
    pow_frames = (mag_spec_frames**2) / mag_spec_frames.shape[1]

    # filter signal with Mel-filter
    frames_envelop, hz_freq = mel_filter(pow_frames, 20, 3800, 40, fs)

    # log of signal
    log_pow_frames = np.log(frames_envelop)

    # perform DCT
    mfcc = discrete_cos_transformation(log_pow_frames)

    # liftering of mffc
    mfcc = sin_liftering(mfcc)

    # take just lwo frequencies mfcc
    mfcc = mfcc[:, 1:13]

    return mfcc