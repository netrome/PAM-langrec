import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from model import SoftmaxClassifier
import sys
from utils import get_slices, meta_to_onehot

# get feature slices and meta data
n, slices, meta = get_slices(sys.argv[1])
onehot, targets = meta_to_onehot(meta)

# create autoencoder
ae = SoftmaxClassifier()
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
out = sess.run("logits:0", feed_dict={"raw_data:0": slices})

# Calculate error rate
print(out.shape)
print(onehot.shape)
classification = np.argmax(out, axis=1)
print(classification)
print("----------------")
print(targets)

acc = np.sum(classification == targets)/len(targets)
print(acc)


