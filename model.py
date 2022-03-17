import tensorflow as tf
import keras
from keras import layers

class OSRCycleGAN(keras.Model):
    def __init__(self, ngf, ndf, training=False):
        super(OSRCycleGAN, self).__init__()

        self.generator_A_B = generator_resnet(ngf, training) # used real image as input
        self.generator_B_A = generator_resnet(ngf, training) # used a fake image as made by fake_A_B as input

        self.discriminator_A = discriminator(ndf, training)
        self.discriminator_B = discriminator(ndf, training)

    def call(self, inputs):
        realA = inputs[0] # low resolution 1번
        realB = inputs[1] # high resolution 1번
        '''
        ┌>>  realA -> fakeB -> reconA ──┐
        │        genAtoB  genBtoA       │
        │        discriB  discriA       │
        └──  reconB <- fakeA <- realB <<┘
        '''

        # A -> G(A) -> F(G(A)) -> B'
        fakeB = self.generator_A_B(realA)
        reconA = self.generator_B_A(fakeB)

        # B - > F(B) -> G(F(B)) -> A'
        fakeA = self.generator_B_A(realB)
        reconB = self.generator_A_B(fakeA)

        discriminate_B = self.discriminator_B(fakeB)
        discriminate_A = self.discriminator_A(fakeA)

        discriminateGT_B = self.discriminator_B(realB)
        discriminateGT_A = self.discriminator_A(realA)

        return fakeB, fakeA, reconB, reconA, discriminate_B, discriminate_A, discriminateGT_B, discriminateGT_A

class discriminator(keras.Model):
    def __init__(self, n_out, training):
        super(discriminator, self).__init__()
        self.n_out = n_out
        self.h0 = Conv2D(self.n_out, kernel_size=4, strides=2, padding='same', batch_norm=False, lrelu=True, training=training)
        self.h1 = Conv2D(self.n_out*2, kernel_size=4, strides=2, padding='same', batch_norm=True, lrelu=True, training=training)
        self.h2 = Conv2D(self.n_out*4, kernel_size=4, strides=2, padding='same', batch_norm=True, lrelu=True, training=training)
        self.h3 = Conv2D(self.n_out*8, kernel_size=4, strides=2, padding='same', batch_norm=True, lrelu=True, training=training)
        self.h4 = Conv2D(1, kernel_size=4, strides=1, padding='same', lrelu=False, batch_norm=False, training=training)

    def call(self, inputs):
        x = self.h0(inputs)
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        # x = self.flatten(x)

        return x
# def generator_resnet(self, x, n_out):

class generator_resnet(keras.Model):
    def __init__(self, n_out, training):
        super(generator_resnet, self).__init__()
        self.n_out = n_out
        self.relu = layers.ReLU()
        self.c1 = Conv2D(self.n_out, kernel_size=7, strides=1, padding='valid', batch_norm=True, lrelu=False, training=training)
        self.c2 = Conv2D(self.n_out*2, kernel_size=3, strides=2, padding='same', batch_norm=True, lrelu=False, training=training)
        self.c3 = Conv2D(self.n_out*4, kernel_size=3, strides=2, padding='same', batch_norm=True, lrelu=False, training=training)
        self.r1 = ResidualConv2D(self.n_out*4)
        self.r2 = ResidualConv2D(self.n_out*4)
        self.r3 = ResidualConv2D(self.n_out*4)
        self.r4 = ResidualConv2D(self.n_out*4)
        self.r5 = ResidualConv2D(self.n_out*4)
        self.r6 = ResidualConv2D(self.n_out*4)

        self.d1 = Deconv2D(self.n_out*2)
        self.d2 = Deconv2D(self.n_out)
        self.pred = Conv2D(3, kernel_size=7, strides=1, padding='valid', batch_norm=False, lrelu=False, training=self.training)

    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        x = self.relu(self.c1(x, training=self.training))
        x = self.relu(self.c2(x, training=self.training))
        x = self.relu(self.c3(x, training=self.training))

        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)

        x = self.d1(x)
        x = self.d2(x)
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = keras.activations.tanh(self.pred(x))

        return pred

class Conv2D(layers.Layer):
    def __init__(self, out, kernel_size, strides, padding, batch_norm, lrelu, training):
        super(Conv2D, self).__init__()
        self.conv2d = layers.Conv2D(filters=out, kernel_size=kernel_size, strides=strides, padding=padding)
        self.batch_norm = batch_norm
        self.lrelu = lrelu
        self.training = training
        if self.batch_norm == True: self.batch_normalization = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, scale=True, trainable=self.training)
        if self.lrelu == True: self.leaky_relu = layers.LeakyReLU(alpha=0.2)
    def call(self, inputs):
        x = self.conv2d(inputs)
        if self.batch_norm == True: x = self.batch_normalization(x, self.training)
        if self.lrelu == True: x = self.leaky_relu(x)
        return x

class Dense(layers.Layer):
    def __init__(self, units, input_dim, activation=None):
        super(Dense, self).__init__()
        self.dense = layers.Dense(units, activation)

    def call(self, inputs):
        return self.dense(inputs)

# class Flatten(layers.Layer):
#     def __init__(self):
#         super(Flatten, self).__init__()
#         self.flatten = layers.Flatten()
#     def call(self, inputs):
#         return self.flatten(inputs)

class ResidualConv2D(layers.Layer):
    def __init__(self, out, kernel_size=3, strides=1, padding='valid', training=False):
        super(ResidualConv2D, self).__init__()
        self.training = training
        self.p = int(kernel_size-1/2)
        self.conv2d = layers.Conv2D(filters=out, kernel_size=kernel_size, strides=strides, padding=padding) # padding='VALID'
        self.batch_norm = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, scale=True, trainable=self.training)
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [self.p, self.p], [self.p, self.p], [0, 0]], "REFLECT")
        x = self.relu(self.batch_norm(self.conv2d(x), self.training))
        x = tf.pad(x, [[0, 0], [self.p, self.p], [self.p, self.p], [0, 0]], "REFLECT")
        x = self.batch_norm(self.conv2d(x), self.training)
        return x + inputs

class Deconv2D(layers.Layer):
    def __init__(self, out=64, kernel_size=4, strides=2, padding='same', training=False):
        super(Deconv2D, self).__init__()
        self.training = training
        self.kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
        self.deconv2d = layers.Conv2DTranspose(filters=out, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, kernel_initializer=self.kernel_initializer)
        self.batch_norm = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, scale=True, trainable=self.training)
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.deconv2d(inputs)
        x = self.relu(self.batch_norm(x, self.training))

        return x
