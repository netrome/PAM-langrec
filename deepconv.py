import tensorflow as tf
import numpy as np

class DeepConvEncoder:
    """ Convolutional Autoencoder class
    """
    
    def __init__(self, image_dims=[100, 26], bottleneck_dim=40):
        """ Sets hyper-parameters

        Input:
            image_dims: image dimensions (default [100, 26])
            bottleneck_dim: dimension of bottleneck layer (default 40)
        """
        self.name = "DeepConv_model"
        self.image_dims = image_dims
        self.bottleneck_dim = bottleneck_dim
    
    def build_model(self):
        """ Builds model graph
        """
        self.images = tf.placeholder(tf.float32, [None] + self.image_dims, name="raw_data")

        self.batch_size = tf.shape(self.images)[0]

        self.encoder = self.encoder(self.images)
        self.decoder = self.decoder(self.encoder)

        self.saver = tf.train.Saver()

 
    def train(self):
        """ Builds training graph
        """
        #err = tf.reduce_mean(tf.abs(self.decoder - self.images), name="err")
        err = tf.reduce_mean(tf.pow(self.decoder - self.images, 2), name="err")
        train_step = tf.train.AdamOptimizer().minimize(err, name="train_step")

        # Add summary scalar for tensor board
        tf.summary.scalar("reduced_abs_err", err)

        return train_step

    
    def encoder(self, images):
        """ Builds encoder graph
        """
        x0 = tf.reshape(images, [self.batch_size] + self.image_dims + [1])

        # First convolutional layer
        W1 = tf.Variable(tf.truncated_normal([7, 7, 1, 12], stddev=0.01))
        x1 = tf.nn.conv2d(x0, W1, [1, 3, 3, 1], padding="SAME")
        h1 = tf.nn.relu(x1)

        # Second convolutional layer
        W2 = tf.Variable(tf.truncated_normal([5, 5, 12, 24], stddev=0.01))
        x2 = tf.nn.conv2d(h1, W2, [1, 2, 2, 1], padding="SAME")
        h2 = tf.nn.relu(x2)

        # Third convolutional layer
        W3 = tf.Variable(tf.truncated_normal([3, 3, 24, 24], stddev=0.01))
        x3 = tf.nn.conv2d(h2, W3, [1, 1, 1, 1], padding="SAME")
        h3 = tf.nn.relu(x3)

        # Fully connected layer
        self.flat_len = np.prod(h3.get_shape().as_list()[1:])
        flat = tf.reshape(h3, [tf.shape(h1)[0], self.flat_len])
        Wf = tf.Variable(tf.truncated_normal([flat.get_shape().as_list()[1], self.bottleneck_dim], stddev=0.01))
        bf = tf.Variable(tf.truncated_normal([self.bottleneck_dim], stddev=0.01))
        xf = tf.nn.xw_plus_b(flat, Wf, bf, name="bottleneck")
        hf = tf.nn.relu(xf)

        self.shapes = [tf.shape(x0), tf.shape(x1), tf.shape(x2), tf.shape(x3)]

        return hf


    def decoder(self, bottleneck):
        """ Builds decoder graph
        """

        # Linear upsampling
        Wf = tf.Variable(tf.truncated_normal([self.bottleneck_dim, self.flat_len], stddev=0.01))
        bf = tf.Variable(tf.truncated_normal([self.flat_len], stddev=0.01))
        xf = tf.nn.xw_plus_b(bottleneck, Wf, bf)
        hf = tf.nn.relu(xf)
        cuboided = tf.reshape(hf, self.shapes[-1])

        # First deconv
        W1 = tf.Variable(tf.truncated_normal([3, 3, 24, 24], stddev=0.01))
        x1 = tf.nn.conv2d_transpose(cuboided, W1, self.shapes[-2], [1, 1, 1, 1])
        h1 = tf.nn.relu(x1)

        # Second deconv
        W2 = tf.Variable(tf.truncated_normal([5, 5, 12, 24], stddev=0.01))
        x2 = tf.nn.conv2d_transpose(h1, W2, self.shapes[-3], [1, 2, 2, 1])
        h2 = tf.nn.relu(x2)
        
        # Third deconv
        W3 = tf.Variable(tf.truncated_normal([7, 7, 1, 12], stddev=0.01))
        x3 = tf.nn.conv2d_transpose(h2, W3, self.shapes[-4], [1, 3, 3, 1])
        h3 = tf.nn.relu(x3)
        
        # Reshape (reduce dimension)
        return tf.reshape(h3, [self.batch_size] + self.image_dims, name="raw_out")

    def save(self, sess, iters, name=""):
        """ Saves tensorflow graph
        """
        path = "./saved_models/model{1}{0}{2}.ckpt".format(iters, self.name, name.split("/")[-1])
        self.saver.save(sess, path)
        return path


    def load(self, sess, iters, name=""):
        """ Loads tensorflow graph
        """
        path = "./saved_models/model{1}{0}{2}.ckpt".format(iters, self.name, name.split("/")[-1])
        self.saver.restore(sess, path)

