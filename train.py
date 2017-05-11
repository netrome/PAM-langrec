import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from model import Autoencoder
from conv import ConvEncoder
from deepconv import DeepConvEncoder
import sys
import os
import csv
from utils import get_slices, images_to_sprite
import scipy.misc

# Get feature slices and meta data
n, slices, meta = get_slices(sys.argv[1])

# Save tsv metadata
meta_writer = csv.writer(open("logs/meta.tsv", "w"), delimiter="\t")
meta_writer.writerows(meta)

# Create autoencoder
ae = DeepConvEncoder()
ae.build_model()
ae.train()

# Add embeddings
projs = tf.Variable(tf.truncated_normal([n, ae.bottleneck_dim]), name="projections")
projs_ass = tf.assign(projs, ae.encoder)
proj_saver = tf.train.Saver([projs])

# Straight from tensorflow
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = projs.name
embedding.metadata_path = "logs/meta.tsv"
if "sprite" in sys.argv:
    sprite = images_to_sprite(slices)
    sprite_path =sys.argv[1].replace("eval_data", "sprites") + "sprite.png"
    scipy.misc.imsave(sprite_path, sprite)
    embedding.sprite.image_path = sprite_path
    embedding.sprite.single_image_dim.extend([26, 100])

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
    patterns = slices[idx]

    for j in range(int(np.floor(n/batch_size))):
        pattern = patterns[j * batch_size : (j + 1) * batch_size] 
        sess.run("train_step", feed_dict={"raw_data:0": pattern})

    if i%10 == 0 and "log" in sys.argv:
        m, tr_err, _ = sess.run([merged, "err:0", projs_ass], feed_dict={"raw_data:0": slices})
        train_writer.add_summary(m, i)
        projector.visualize_embeddings(train_writer, config)
        ae.saver.save(sess, "/tmp/tf/model.cpkt", global_step=i)
        proj_saver.save(sess, "/tmp/tf/proj.cpkt", global_step=i)

        # Save in numpy format
        train_err.append(tr_err)
        np.save("logs/train_err", train_err)
    print(i)

# Save trained model
save_path = ae.save(sess, iters, sys.argv[1])
print("Saved model in {0}".format(save_path))

