import numpy as np


def framing(signal: np.array, fs: float, frame_size: float, frame_step: float):
    """
    Frame input signal in frames with frame_size and frame_step defined in seconds.

    :param signal: Input signal
    :param fs: Sampling frequency
    :param frame_size: Size of each frame in seconds
    :param frame_step: Steps between frames in seconds
    :return: Frames of signal
    """

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
    """
    Apply hamming window to each frame.

    :param frames: Input signal frames
    :return: Windowed frames
    """

    window_length = frames.shape[1]
    n = np.arange(0, window_length)

    # set window coefficients
    h = 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_length - 1))

    # perform windowing
    frames *= h

    return frames


def stft(y: np.array, fs: float, n_fft: int, frame_size: float, frame_step: float):
    """
    Creates spectrogram of input signal y.

    :param y: Input signal
    :param fs: Sampling frequency
    :param n_fft: Num of points to calculate FFT
    :param frame_size: Size of frame in seconds
    :param frame_step: Steps between frames in seconds
    :return: Spectrogram frames
    """

    # framing
    frames = framing(y, fs, frame_size, frame_step)

    # window the frames with hamming window
    frames = hamming(frames)

    # create magnitude spectrogram
    spec_frames = np.fft.fft(frames, n=n_fft, axis=1)

    # take one spectrogram side
    spec_frames = spec_frames[:, 0: int(n_fft / 2 + 1)]

    return spec_frames


def mel_filter(frames, f_min, f_max, n_mels, fs):
    """
    Applies Mel filter to input spectrogram frames.

    :param frames: Input spectrogram frames
    :param f_min: Minimum frequency of Mel filter
    :param f_max: Maximum frequency of Mel filter
    :param n_mels: Number of filters in Mel filter
    :param fs: Sampling frequency
    :return: Mel filtered frames, start-end frequencies of each Mel filter
    """

    n_fft = frames.shape[1] - 1
    # convert Hz to Mel frequency
    mel_lf = 2595 * np.log10(1 + f_min / 700)
    mel_hf = 2595 * np.log10(1 + f_max / 700)

    mel_points = np.linspace(mel_lf, mel_hf, n_mels + 2)

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

        for k in np.arange(left_f, center_f + 1):
            f_bank[i - 1, k] = (k - left_f) / (center_f - left_f)

        for k in np.arange(center_f, right_f + 1):
            f_bank[i - 1, k] = (-k + right_f) / (-center_f + right_f)

        # scale filter bank by its width
        f_bank[i - 1] /= (hz_points[i] - hz_points[i-1])

    # filter frames
    filtered_frames = np.dot(frames, f_bank.T)

    # correct 0 values
    filtered_frames += np.finfo(float).eps

    return filtered_frames, hz_points


def signal_power_to_db(power_frames, min_amp=1e-10, top_db=80):
    """
    Covert power spectrum amplitude to dB.

    :param power_frames: Power spectrum frames
    :param min_amp: Minimum amplitude
    :param top_db: Max value ib dB
    :return: Power dB frames
    """
    log_spec = 10.0 * np.log10(np.maximum(min_amp, power_frames))
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def discrete_cos_transformation(frames: np.array):
    """
    Applies DCT to input frames

    :param frames: Input frames
    :return: DCT frames
    """

    rows, cols = frames.shape

    N = cols
    n = np.arange(1, N + 1)

    weights = np.zeros((N, N))
    for k in np.arange(0, N):
        weights[:, k] = np.cos(np.pi * (n - 1 / 2) * k / N)

    dct_signal = np.sqrt(2 / N) * np.dot(frames, weights)

    return dct_signal


def sin_liftering(mfcc: np.array):
    """
    Applies sin liftering to input mfcc frames

    :param mfcc: Input mfcc frames
    :return: Sin liftered frames
    """

    mfcc_lift = np.zeros(mfcc.shape)

    # create lifting window
    n = np.arange(1, mfcc_lift.shape[1] + 1)
    D = 22
    w = 1 + (D / 2) * np.sin(np.pi * n / D)

    # lift coefficients
    mfcc_lift = mfcc * w

    return mfcc_lift
