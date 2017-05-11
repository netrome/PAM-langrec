import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from model import Autoencoder
from conv1 import ConvEncoder
from conv2 import ConvEncoder2
from conv3 import ConvEncoder3
from vae import VAE
from convskip import ConvSkip
from fullskip import FullSkip
import sys

path = sys.argv[1]
img = image.imread(path, mode="RGB").astype(np.float)
img = img.reshape([1] + list(img.shape))

# create autoencoder
ae = ConvSkip()
ae.build_model()
ae.train()

iters = 700
if len(sys.argv) > 1:
    iters = sys.argv[2]

# Restore
sess = tf.Session()
ae.load(sess, iters)
print()
print()
print("------------------------------")

out = sess.run("raw_out:0", feed_dict={"raw_data:0": img})
plt.imshow(out[0])
plt.show()
