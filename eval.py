import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from model import Autoencoder
from conv import ConvEncoder
from deepconv import DeepConvEncoder
import sys
from utils import get_slices

# get feature slices and meta data
n, slices, meta = get_slices(sys.argv[1])

# create autoencoder
ae = DeepConvEncoder()
ae.build_model()
ae.train()

iters = 700
if len(sys.argv) > 1:
    iters = sys.argv[2]

# Restore
sess = tf.Session()
ae.load(sess, iters, sys.argv[1])
print()
print()
print("------------------------------")

# Test the model
idx = [0, 1, 2, 3, 4, -1, -2, -3, -4]
samples = slices[idx]
out = sess.run("raw_out:0", feed_dict={"raw_data:0": samples})

# Plot some samples

plt.figure()

for i in range(out.shape[0]):
    #pueh = (out[i] - np.min(out[i])) / (np.max(out[i]) - np.min(out[i]) + 1E-6)
    plt.subplot(2, out.shape[0], i + 1)
    plt.pcolormesh(out[i])
    plt.subplot(2, out.shape[0], i + out.shape[0] + 1)
    plt.pcolormesh(samples[idx[i]])
    #print("Scaled with value: ", np.max(out[i]) - np.min(out[i]) + 1E-6)

plt.show()
