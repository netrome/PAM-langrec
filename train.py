import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from model import SoftmaxClassifier
from conv import ConvClassifier
import sys
import os
import csv
from utils import get_slices, images_to_sprite, meta_to_onehot
import scipy.misc

# Get feature slices and meta data
n, slices, meta = get_slices(sys.argv[1])

onehot, _ = meta_to_onehot(meta)

n = int(np.floor(slices.shape[0] * 9/10))
tr_slices = slices[:n]
tr_onehot = onehot[:n]
val_slices = slices[n:]
val_onehot = onehot[n:]

print(onehot)
print(onehot.shape)
print(slices.shape)

# Save tsv metadata
meta_writer = csv.writer(open("logs/meta.tsv", "w"), delimiter="\t")
meta_writer.writerows(meta)

# Create autoencoder
ae = ConvClassifier()
ae.build_model()
ae.train()

# Do the training
init = tf.global_variables_initializer()
sess = tf.Session()

# File writer for tensorboard
if "board" in sys.argv:
    os.system("rm -rf /tmp/tf/")
    os.system("killall tensorboard")
    os.system("tensorboard --logdir /tmp/tf/ --port 6006 &")

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/tmp/tf/train/", sess.graph)
val_writer = tf.summary.FileWriter("/tmp/tf/val/", sess.graph) 

sess.run(init)
print()
print()
print("------------------------------")

train_err = []
val_err = []

iters = 20
if len(sys.argv) > 2:
    iters = int(sys.argv[2])

if len(sys.argv) > 3:
    batch_size = int(sys.argv[3])

for i in range(iters): 
    idx = np.random.permutation(n)
    patterns = tr_slices[idx]
    targets = tr_onehot[idx]

    for j in range(int(np.floor(n/batch_size))):
        pattern = patterns[j * batch_size : (j + 1) * batch_size] 
        target = targets[j * batch_size : (j + 1) * batch_size] 
        sess.run("train_step", feed_dict={"raw_data:0": pattern, "targets:0": target})

    if i%10 == 0 and "log" in sys.argv:
        m, tr_err = sess.run([merged, "err:0"], feed_dict={"raw_data:0": tr_slices[:100], "targets:0": tr_onehot[:100]})
        train_writer.add_summary(m, i)
        m, va_err = sess.run([merged, "err:0"], feed_dict={"raw_data:0": val_slices, "targets:0": val_onehot})
        val_writer.add_summary(m, i)
        ae.saver.save(sess, "/tmp/tf/model.cpkt", global_step=i)

        # Save in numpy format
        train_err.append(tr_err)
        val_err.append(va_err)
        np.save("logs/train_err", train_err)
        np.save("logs/val_err", val_err)
    print(i)

# Save trained model
save_path = ae.save(sess, iters, sys.argv[1])
print("Saved model in {0}".format(save_path))

