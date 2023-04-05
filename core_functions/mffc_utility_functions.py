import numpy as np


def preemphasis_filter(signal: np.array, coeff: float):
    signal_1 = np.roll(signal, shift=1)
    signal_1[0] = 0
    return signal - coeff * signal_1


def framing(signal: np.array, frame_size: float, frame_step: float, fs: float):
    # init variables
    frame_length = np.round(frame_size * fs).astype(int)
    frame_step = np.round(frame_step * fs).astype(int)
    signal_length = signal.shape[0]

    # calculate number of frames
    n_frames = np.ceil(abs(signal_length - frame_length) / frame_step).astype(int)

    # pad signal that all frames have equal number of samples
    pad_signal_length = int(n_frames * frame_step + frame_length)
    zeros_pad = np.zeros((1, pad_signal_length - signal_length))
    pad_signal = np.concatenate((signal.reshape((1, -1)), zeros_pad), axis=1).reshape(-1)

    # extract frames
    frames = np.zeros((n_frames, frame_length))
    indices = np.arange(0, frame_length)

    for i in np.arange(0, n_frames):
        offset = i * frame_step
        frames[i] = pad_signal[(indices + offset)]

    return frames


def hamming(frames: np.array):
    window_length = frames.shape[1]
    n = np.arange(0, window_length)

    # set window coefficients
    h = 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_length - 1))

    # perform windowing
    frames *= h

    return frames


def mel_filter(frames, f_min, f_max, n_mels, fs):

    n_fft = frames.shape[1] - 1
    # convert Hz to Mel frequency
    mel_lf = 2595 * np.log10(1 + f_min / 700)
    mel_hf = 2595 * np.log10(1 + f_max / 700)

    mel_points = np.linspace(mel_lf, mel_hf, n_fft + 2)

    # convert back Mel to Hz
    hz_points = 700 * (np.power(10, mel_points / 2595) - 1)

    fft_bank_bin = np.floor((n_fft + 1) * hz_points / (fs / 2))
    fft_bank_bin[-1] = n_fft

    # create filter banks
    f_bank = np.zeros((n_mels, n_fft + 1))
    for i in np.arange(1, n_mels + 1):
        left_f = int(fft_bank_bin[i - 1])
        center_f = int(fft_bank_bin[i])
        right_f = int(fft_bank_bin[i + 1])

        for k in np.arange(left_f, center_f+1):
            f_bank[i - 1, k] = (k - left_f) / (center_f - left_f)

        for k in np.arange(center_f, right_f+1):
            f_bank[i - 1, k] = (-k + right_f) / (-center_f + right_f)

    # filter frames
    filtered_frames = np.dot(frames, f_bank.T)
    # correct 0 values
    filtered_frames += np.finfo(float).eps

    return filtered_frames


def discrete_cos_transformation(frames: np.array):
    rows, cols = frames.size()
    dct_signal = np.zeros((rows, cols))
    N = cols
    n = np.arange(1, N)
    for i in np.arrange(0, rows):
        signal = frames[i, :]
        X = np.zeros((1, N))
        for k in np.arrage(1, N):
            X[k] = np.sum(signal * np.cos(np.pi * (n - 1 / 2) * (k - 1) / N))
            X[k] = np.sqrt(2 / N) * X[k];

        dct_signal[i] = X

    return dct_signal


def sin_liftering(mfcc: np.array):
    mfcc_lift = np.zeros(mfcc.size())

    # create lifting window
    n = np.arange(1, mfcc_lift.shape[1] + 1)
    D = 22
    w = 1 + (D / 2) * np.sin(np.pi * n / D)
    for i in np.arange(0, mfcc.shape[0]):
        mfcc_lift[i] *= w

    return mfcc_lift
