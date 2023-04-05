import numpy as np
from matplotlib import pyplot as plt

from core_functions.mffc_utility_functions import preemphasis_filter, framing, hamming, mel_filter, \
    discrete_cos_transformation, sin_liftering


def extract_mffc_feature(y: np.array, fs: float):
    # pre emphasis filter
    filter_coeff = 0.95
    y = preemphasis_filter(y, filter_coeff)

    # framing
    frame_size = 0.025
    frame_step = 0.01
    frames = framing(y, frame_size, frame_step, fs)

    # window the frames with hamming window
    frames = hamming(frames)

    # create magnitude spectrogram
    n_fft = 512
    mag_frames = abs(np.fft.fft(frames, n=n_fft, axis=1))
    mag_frames = mag_frames[:, 1: int(n_fft / 2 + 1)]

    # create power spectrogram
    pow_frames = (mag_frames**2) / frames.shape[1]

    # filter signal with Mel-filter
    frames_envelop, hz_freq = mel_filter(pow_frames, 20, 3800, 40, fs)

    # log of signal
    log_pow_frames = np.log(frames_envelop)

    # perform DCT
    mfcc = discrete_cos_transformation(log_pow_frames)

    # liftering of mffc
    mfcc = sin_liftering(mfcc)

    return mfcc