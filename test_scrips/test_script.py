import numpy as np
import scipy.io.wavfile as sci_wav # Open wav files
from matplotlib import pyplot as plt

from core_functions.mffc_feature_extraction import extract_mffc_feature


# Fcn to read wav files from root directory
def read_wav_files(root_dir, wav_files):
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(root_dir + f)[1] for f in wav_files]


location = "./TV1.wav"
# read sound data
fs, y = sci_wav.read(location)
y = y[4750:]
mfcc = extract_mffc_feature(y, fs)