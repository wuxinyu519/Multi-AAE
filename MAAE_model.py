import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import regularizers


def make_encoder_model(img_size, num_c, pro, z_dim):
    initializer = tf.keras.initializers.glorot_normal(seed=42)
    inputs = tf.keras.Input(shape=(img_size * img_size * num_c,))
    x = tf.keras.layers.Dense(1000, kernel_initializer=initializer)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(pro)(x)

    x = tf.keras.layers.Dense(1000, kernel_initializer=initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(pro)(x)
    mean = tf.keras.layers.Dense(z_dim, kernel_initializer=initializer)(x)
    stddev = tf.keras.layers.Dense(z_dim, kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=[mean, stddev])
    return model


def make_decoder_model(img_size, num_c, pro, z_dim):
    initializer = tf.keras.initializers.glorot_normal(seed=42)
    encoded = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(1000, kernel_initializer=initializer)(encoded)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(pro)(x)
    x = tf.keras.layers.Dense(1000, kernel_initializer=initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(pro)(x)
    reconstruction = tf.keras.layers.Dense(img_size * img_size * num_c, kernel_initializer=initializer,
                                           activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def make_discriminator_model(pro, z_dim, h_dim, num_layers):
    initializer = tf.keras.initializers.glorot_normal()
    encoded = tf.keras.Input(shape=(z_dim,))
    x = encoded

    for _ in range(num_layers):
        x = tf.keras.layers.Dense(h_dim, kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(pro)(x)

    prediction = tf.keras.layers.Dense(1, kernel_initializer=initializer, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


def make_semi_discriminator_model(pro, z_dim, h_dim, num_layers):
    z_dim=z_dim+10
    initializer = tf.keras.initializers.glorot_normal(seed=42)
    encoded = tf.keras.Input(shape=(z_dim,))
    x = encoded

    for _ in range(num_layers):
        x = tf.keras.layers.Dense(h_dim, kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(pro)(x)

    prediction = tf.keras.layers.Dense(1, kernel_initializer=initializer, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


def make_cnn_encoder_model(img_size, num_c, z_dim):
    initializer = tf.keras.initializers.glorot_normal(seed=42)
    inputs = tf.keras.Input(shape=(img_size, img_size, num_c,))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same'
                               , kernel_initializer=initializer)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same'
                               , kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same'
                               , kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same'
                               , kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(256
                              , kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    mean = tf.keras.layers.Dense(z_dim, kernel_initializer=initializer)(x)
    stddev = tf.keras.layers.Dense(z_dim, kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=[mean, stddev])
    return model


def make_cnn_decoder_model(z_dim, img_size):
    encoded = tf.keras.Input(shape=(z_dim,))
    initializer = tf.keras.initializers.glorot_normal(seed=42)
    # x = tf.keras.layers.Dense(128, kernel_initializer=initializer)(encoded)
    # x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense((img_size // 16) * (img_size // 16) * 64, kernel_initializer=initializer)(encoded)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape(((img_size // 16), (img_size // 16), 64))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1, padding='same',
                                        kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same',
                                        kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same',
                                        kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same',
                                        kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', activation='sigmoid'
                                             , kernel_initializer=initializer)(
        x)

    model = tf.keras.Model(inputs=encoded, outputs=output)
    return model


def autoencoder_loss(inputs, reconstruction):
    return tf.reduce_mean(mse(inputs, reconstruction))


class MAAE(keras.Model):
    def __init__(self, num_of_d, latent_dim, img_size, num_c, std, training_style):
        super(MAAE, self).__init__()
        self.num_of_d = num_of_d
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_c = num_c
        self.training_style = training_style
        self.keep_pro = 0.3
        self.std = std
        self.h_dim = 1000
        if num_c == 1:
            self.encoder = make_encoder_model(self.img_size, self.num_c, self.keep_pro, self.latent_dim)
            self.decoder = make_decoder_model(self.img_size, self.num_c, self.keep_pro, self.latent_dim)
            if training_style == 'semi':
                self.discriminator = [
                    make_semi_discriminator_model(self.keep_pro + (0.5 - self.keep_pro) * i / self.num_of_d,
                                                  self.latent_dim,
                                                  self.h_dim, 2) for i in range(self.num_of_d)]
            else:
                self.discriminator = [
                    make_discriminator_model(self.keep_pro + (0.5 - self.keep_pro) * i / self.num_of_d, self.latent_dim,
                                             self.h_dim, 2) for i in range(self.num_of_d)]


        else:
            # self.encoder = conv_encoder(self.img_size, self.num_c, self.latent_dim)
            self.encoder = make_cnn_encoder_model(self.img_size, self.num_c, self.latent_dim)
            self.decoder = make_cnn_decoder_model(self.latent_dim, self.img_size)
            self.discriminator = [
                make_discriminator_model(self.keep_pro + (0.5 - self.keep_pro) * i / self.num_of_d, self.latent_dim,
                                         self.h_dim,2) for
                i in
                range(self.num_of_d)]
            #different layer
            # self.discriminator = [
            #     make_discriminator_model(0.3, self.latent_dim,
            #                              self.h_dim, i + 1) for i in range(self.num_of_d)]

    def call(self, inputs, training=False):
        if training:
            # Do something during training
            print('Training mode')
        else:
            # Do something during inference
            print('Inference mode')
        z = encode(inputs, training=training)
        reconstructed = self.decode(z, training=training)
        return reconstructed

    def encode(self, inputs, training):
        z_mean, z_log_var = self.encoder(inputs, training=training)
        z = self.sampling(z_mean, z_log_var)
        return z

    def decode(self, z, training):
        reconstructed = self.decoder(z, training=training)
        return reconstructed

    def sampling(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0., stddev=self.std)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
