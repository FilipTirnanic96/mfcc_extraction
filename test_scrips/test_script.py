import numpy as np
import scipy.io.wavfile as sci_wav # Open wav files
from matplotlib import pyplot as plt

from core_functions.mffc_feature_extraction import extract_mffc_feature
from core_functions.mffc_utility_functions import stft


# Fcn to read wav files from root directory
def read_wav_files(root_dir, wav_files):
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(root_dir + f)[1] for f in wav_files]


location = "./TV1.wav"
# read sound data
fs, y = sci_wav.read(location)
y = y[4750:]/ max(np.abs(y[4750:]))
t_lim = y.shape[0] / fs
plt.plot(y)
n_fft = 512
mfcc = extract_mffc_feature(y, fs, n_fft)

time=np.linspace(0, t_lim, mfcc.shape[0])
plt.figure(figsize=(6, 8))
plt.imshow(mfcc.T, extent=[0, 15, 13, 1])
plt.gca().invert_yaxis()
plt.show()


spec = np.log(np.abs(stft(y, fs, n_fft, 0.025, 0.01)))
plt.figure(figsize=(6, 8))
plt.imshow(spec.T, extent=[0, 1, 0, 1])
plt.gca().invert_yaxis()
plt.show()