import os
import sys
import glob
import numpy as np
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import fbank
from scipy.signal import stft
import scipy.io.wavfile as wav

# Get data
place = os.getcwd()
sound_path = place + "/sounds/"

data = glob.glob(os.path.join(sound_path, "*.wav"))

patterns = []

for path in data:
    rate, sig = wav.read(path)
    if "mfcc" in sys.argv:
        feat = mfcc(sig, rate)
    elif "fbank" in sys.argv:
        feat = fbank(sig, rate)[0]
    elif "logfbank" in sys.argv:
        feat = logfbank(sig, rate)
    elif "powspec" in sys.argv:
        feat = stft(sig)[2].transpose()
        feat = np.real(feat * np.conj(feat))
    else:
        raise IndexError("Ge mig ett jävla kommandoradsargument för fan!")
    patterns.append(feat)

patterns = np.array(patterns)

np.save("numpy_features", patterns)
