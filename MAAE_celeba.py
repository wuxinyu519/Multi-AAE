"""
Probabilistic unsupervised adversarial autoencoder.
 We are using:
    - Gaussian distribution as prior and posterior distribution.
    - Dense layers.
    - Cyclic learning rate.
"""
import os
import time
from pathlib import Path
from math import sin, cos, sqrt
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from random import *
from scipy.linalg import sqrtm
import pandas as pd
import keras.backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from numpy.random import randint
from scipy.linalg import sqrtm

from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from skimage.color import gray2rgb
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50

UNIQUE_RUN_ID = 'MAAE_celeba_style'
PROJECT_ROOT = Path.cwd()
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        exit(-1)
# -------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------
output_dir = PROJECT_ROOT / 'outputs'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'MAAE_celeba_style'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'latent_space'
latent_space_dir.mkdir(exist_ok=True)

reconstruction_dir = experiment_dir / 'reconstruction'
reconstruction_dir.mkdir(exist_ok=True)

style_dir = experiment_dir / 'style'
style_dir.mkdir(exist_ok=True)
# Set random seed
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

train_data = PROJECT_ROOT/'celeba' / 'celeba_data' / 'img_align_celeba'
FEATURE_PATH = PROJECT_ROOT/'celeba' / 'celeba_data' / 'list_attr_celeba.csv'
# test_data = PROJECT_ROOT / 'celeba_data' / 'test_data'

# -------------------------------------------------------------------------------------------------------------
# HYPERPARAMETER
# -----------------------------------------------------------------------------------------
img_size = 32
num_c = 3
batch_size = 100
NUM_OF_D = 3
n_samples = len(os.listdir(train_data)[50000])
z_dim = 15
d_dim = 1000
std = 5.
ae_lr = 0.0002
gen_lr = 0.0001
dc_lr = 0.0001
# max_lr = 0.001
# step_size = 2 * np.ceil(n_samples / batch_size)
# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate
# WEIGHT_INIT_STDDEV = 0.02
# step_size = 2 * np.ceil(x_train.shape[0] / batch_size)
global_step = 0
n_epochs = 101
keep_pro = 0.1
n_labels = 2
# -------------------------------------------------------------------------------------------------------------
# Define loss functions
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.
lam = [tf.Variable(tf.constant(0.01))]
# gen_loss_weight=[tf.Variable(1-tf.constant(1.))]
cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()
accuracy = tf.keras.metrics.BinaryAccuracy()
initializer = tf.keras.initializers.glorot_normal(seed=42)
# -------------------------------------------------------------------------------------------------------------
# Loading data
# -------------------------------------------------------------------------------------------------------------
print("Loading data...")

x_train = tf.keras.preprocessing.image_dataset_from_directory(
    train_data,
    batch_size=batch_size,
    label_mode=None,
    image_size=(img_size, img_size),
    shuffle=False
)
for img in x_train:
    plt.imshow(img[0])
    plt.savefig("face.png")
    break
x_train = x_train.take(50000)  # 只获取前 50000 张图像

x_train = x_train.map(lambda x: tf.cast(x, tf.float32) / 255.0)


df = pd.read_csv(FEATURE_PATH, usecols=['Male'], nrows=50000)

df.loc[df['Male'] == -1, 'Male'] = 0
df.loc[df['Male'] == 1, 'Male'] = 1
y_train = df["Male"].values.astype('int32')
print(y_train)
print(x_train)
y_train = tf.data.Dataset.from_tensor_slices((y_train))

# train_dataset = y.shuffle(buffer_size=train_buf)
y_train = y_train.batch(batch_size)

train_dataset = tf.data.Dataset.zip((x_train, y_train))



def save_models(decoder, encoder, discriminator, epoch):
    """ Save models at specific point in time. """
    tf.keras.models.save_model(
        decoder,
        f'./{UNIQUE_RUN_ID}/decoder_{epoch}.model',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    tf.keras.models.save_model(
        encoder,
        f'./{UNIQUE_RUN_ID}/encoder_{epoch}.model',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    for ind in range(NUM_OF_D):
        tf.keras.models.save_model(
            discriminator[ind],
            f'./{UNIQUE_RUN_ID}/discriminator{ind}_{epoch}.model',
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )


def conv_block(inputs, filters, strides):
    """
    Convolutional block in ResNet-18.
    """
    x = tf.keras.layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def identity_block(inputs, filters):
    """
    Identity block in ResNet-18.
    """
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([x, inputs])
    x = tf.keras.layers.ReLU()(x)
    return x


def make_encoder_model():
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

def make_decoder_model():
    encoded = tf.keras.Input(shape=(z_dim+n_labels,))
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

# 定义判别器
def make_discriminator_model(pro):
    inputs = tf.keras.layers.Input(shape=(z_dim,))
    # x = tf.keras.layers.Dense(1024, activation='relu')
    # x = tf.keras.layers.Dropout(pro/4)(x)  # 添加 Dropout 层，设置丢弃率为 0.5
    x = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer=initializer)(inputs)
    x = tf.keras.layers.Dropout(pro)(x)  # 添加 Dropout 层，设置丢弃率为 0.5
    x = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dropout(pro)(x)  # 添加 Dropout 层，设置丢弃率为 0.5

    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs, x)


def autoencoder_loss(inputs, reconstruction):
    return ae_loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output):
    # d_loss = [tf.reduce_mean(-tf.math.log(real_output[ind]) - tf.math.log(1 - fake_output[ind])) for ind in range(NUM_OF_D)]
    loss_real = [cross_entropy(tf.ones_like(real_output[ind]), real_output[ind]) for ind in range(NUM_OF_D)]
    loss_fake = [cross_entropy(tf.zeros_like(fake_output[ind]), fake_output[ind]) for ind in range(NUM_OF_D)]
    d_loss = [loss_real[i] + loss_fake[i] for i in range(NUM_OF_D)]
    return d_loss
    # return tf.reduce_mean(-tf.math.log(real_output) - tf.math.log(1 - fake_output))


def mix_pre(G_losses):
    used_l = tf.nn.softplus(lam)
    weights = tf.exp(used_l * G_losses)

    denom = tf.reduce_sum(weights)
    weights = tf.math.divide(weights, denom)
    print('lam_weights shape', weights.shape)
    g_loss = tf.reduce_sum(weights * G_losses)
    return g_loss, used_l


def generator_loss(fake_output):
    G_losses = [cross_entropy(tf.ones_like(fake_output[ind]), fake_output[ind]) for ind in range(NUM_OF_D)]
    gen_loss, used_l = mix_pre(G_losses, lam)
    g_loss = gen_loss - 0.001 * used_l
    return g_loss, used_l
    # return tf.reduce_max(G_losses)


def reparameterization(z_mean, z_std):
    # Probabilistic with Gaussian posterior distribution
    epsilon = tf.random.normal(shape=z_mean.shape, mean=0., stddev=std)
    z = z_mean + tf.exp(0.5 * z_std) * epsilon
    return z


def reparameterization_2(z_mean_std):
    # Probabilistic with Gaussian posterior distribution
    epsilon = [tf.random.normal(shape=z_mean_std[1][0].shape, mean=0., stddev=std) for _ in range(NUM_OF_D)]

    z = [z_mean_std[ind][0] + tf.exp(0.5 * z_mean_std[ind][1]) * epsilon[ind] for ind in range(NUM_OF_D)]
    return z


# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate
# ae_lr = 0.0002
# dc_lr = 0.0001
# gen_lr = 0.0001


# Define optimizers
ae_optimizer = tf.keras.optimizers.Adam(lr=ae_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=dc_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=gen_lr)
encoder = make_encoder_model()
decoder = make_decoder_model()
discriminator = [make_discriminator_model(0.3 + 0.2 * i / NUM_OF_D) for i in range(NUM_OF_D)]
# encoder.summary()
# decoder.summary()
# discriminator[0].summary()


@tf.function
def train_step(batch_x,batch_y):
    with tf.GradientTape(persistent=True) as ae_tape:
        z_mean, z_std = encoder(batch_x, training=True)
        z = reparameterization(z_mean, z_std)

        batch_y_one_hot = tf.one_hot(batch_y, n_labels)
        z_label = tf.concat(
            [z, batch_y_one_hot], axis=1
        )
        decoder_output = decoder(z_label, training=True)
        # Autoencoder loss

        ae_loss = autoencoder_loss(batch_x, decoder_output)

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # Discriminator
    with tf.GradientTape(persistent=True) as tape:
        real_distribution = [tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=std) for _ in range(NUM_OF_D)]


        z_mean, z_std = encoder(batch_x, training=True)
        z = reparameterization(z_mean, z_std)
        dc_real = [discriminator[ind](real_distribution[ind], training=True) for ind in
                   range(NUM_OF_D)]
        dc_fake = [discriminator[ind](z, training=True) for ind in
                   range(NUM_OF_D)]
        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake)
        # Discriminator Acc
        dc_acc = [accuracy(tf.concat([tf.ones_like(dc_real[ind]), tf.zeros_like(dc_fake[ind])], axis=0),
                           tf.concat([dc_real[ind], dc_fake[ind]], axis=0)) for ind in range(NUM_OF_D)]

        # Generator loss
        gen_loss, used_l = generator_loss(dc_fake)
        sum_loss = gen_loss - 0.001 * used_l

    dc_grads_discriminator = [tape.gradient(dc_loss[ind], discriminator[ind].trainable_variables) for ind in
                              range(NUM_OF_D)]
    dc_grads_generator = tape.gradient(sum_loss, encoder.trainable_variables + lam)

    for ind in range(NUM_OF_D):
        dc_optimizer.apply_gradients(zip(dc_grads_discriminator[ind], discriminator[ind].trainable_variables))

    gen_optimizer.apply_gradients(zip(dc_grads_generator, encoder.trainable_variables + lam))

    del tape, ae_tape

    return ae_loss, dc_loss, dc_acc, gen_loss, used_l


# -------------------------------------------------------------------------------------------------------------
# Training loop
if not os.path.exists(f'./cruves/{UNIQUE_RUN_ID}'):
    os.mkdir(f'./cruves/{UNIQUE_RUN_ID}')
writer = tf.summary.create_file_writer(f'./cruves/{UNIQUE_RUN_ID}/')
with writer.as_default():
    # real_images = np.array(generate_real_images(x_test_fid, 10000))
    enc_std = []
    enc_dec_std = []
    all_std = []
    for epoch in range(n_epochs):
        start = time.time()

        epoch_ae_loss_avg = tf.metrics.Mean()
        epoch_dc_loss_avg = tf.metrics.Mean()
        epoch_dc_acc_avg = tf.metrics.Mean()
        epoch_gen_loss_avg = tf.metrics.Mean()
        for batch, (batch_x,batch_y) in enumerate(train_dataset):
            # print(batch_y)
            # -------------------------------------------------------------------------------------------------------------
            # Calculate cyclic learning rate
            global_step = global_step + 1
            # 调整输入图片大小
            # batch_x = tf.image.resize(batch_x, (224, 224))
            ae_loss, dc_loss, dc_acc, gen_loss, used_l = train_step(batch_x,batch_y)

            epoch_ae_loss_avg(ae_loss)
            # epoch_dc_loss_avg(dc_loss)
            # epoch_dc_acc_avg(dc_acc)
            epoch_gen_loss_avg(gen_loss)

            # caculate the Standard Deviation of the encoder loss(recon+adversarial) after each 100 iterations.
            enc_std.append(np.mean(gen_loss))
            enc_dec_std.append(np.mean(ae_loss))
            all_std.append(np.mean(gen_loss) + np.mean(ae_loss))
            if (batch + 1) % 100 == 0:
                tf.summary.scalar("enc_std", np.mean(np.std(enc_std)), global_step)
                tf.summary.scalar("enc_dec_std", np.mean(np.std(enc_dec_std)), global_step)
                tf.summary.scalar("all_std", np.mean(np.std(all_std)), global_step)
                enc_std.clear()
                enc_dec_std.clear()
                all_std.clear()
            if global_step % 10 == 0:
                tf.summary.scalar("ae_loss", np.mean(epoch_ae_loss_avg.result()), global_step)
                tf.summary.scalar("dc_loss", np.mean(dc_loss), global_step)
                # [tf.summary.scalar("dc_loss%d" % ind, np.mean(dc_loss[ind]), global_step) for ind in range(NUM_OF_D)]
                tf.summary.scalar("gen_loss", np.mean(epoch_gen_loss_avg.result()), global_step)
                # [tf.summary.scalar("gen_loss%d" % ind, np.mean(G_losses[ind]), global_step) for ind in range(NUM_OF_D)]
                tf.summary.scalar("dc_acc", np.mean(dc_acc), global_step)
                tf.summary.scalar("used_l", np.mean(used_l), global_step)
            # if (global_step % 200 == 0) and (epoch < 3):
            #     # Sampling
            #     num = 5
            #     """ Generate subplots with generated examples. """
            #     z = tf.random.normal([num * num, z_dim], mean=0.0, stddev=std)
            #     images = decoder(z, training=False)
            #     plt.figure(figsize=(5, 5), dpi=300)
            #     for i in range(num * num):
            #         # Get image and reshape
            #         image = images[i]
            #         image = np.reshape(image, (img_size, img_size, num_c))
            #         # Plot
            #         plt.subplot(num, num, i + 1)
            #         plt.imshow(image[:, :, :])
            #         plt.axis('off')
            #     plt.savefig(style_dir / ('iter_%d.png' % global_step))

        epoch_time = time.time() - start
        print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
              .format(epoch, epoch_time,
                      epoch_time * (n_epochs - epoch),
                      epoch_ae_loss_avg.result(),
                      np.mean(dc_loss),
                      np.mean(dc_acc),
                      epoch_gen_loss_avg.result()

                      ))
        if epoch % 10 == 0:
            save_models(decoder, encoder, discriminator, epoch)
            # Conditioned Sampling

        # -------------------------------------------------------------------------------------------------------------
        if epoch % 1 == 0:

            # Reconstruction
            # n_digits = 25  # how many digits we will display
            # for batch, (batch_x) in enumerate(x_test):
            #     test_images = batch_x[:n_digits]
            #     break
            # z, z_std = encoder(test_images, training=False)
            # x_test_decoded = decoder(z, training=False)
            # x_test_decoded = np.reshape(x_test_decoded, [-1, img_size, img_size, num_c])
            # fig = plt.figure(figsize=(5, 5), dpi=300)
            # for i in range(n_digits):
            #     # Plot
            #     plt.subplot(5, 5, i + 1)
            #     plt.imshow(x_test_decoded[i][:, :, :])
            #     plt.axis('off')
            # plt.savefig(reconstruction_dir / ('epoch_%d.png' % epoch))
            # Sampling
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np

            # 设置图片的大小
            # fig = plt.figure(figsize=(20, 1))

            nx, ny = 10, 2
            # Sampling
            plt.subplot()
            fig = plt.figure(figsize=(nx, ny), dpi=300)
            i = 0
            for t in range(nx):
                z = tf.random.normal([1, z_dim], mean=0, stddev=5)
                for r in range(ny):
                    label = np.random.randint(r, r + 1, size=[1])
                    label_sample_one_hot = tf.one_hot(label, n_labels)

                    dec_input = tf.concat(
                        [z, label_sample_one_hot], axis=1
                    )
                    x = decoder(dec_input, training=False).numpy()
                    ax = plt.subplot(t+1,r+1,i+1)
                    i += 1
                    img = np.array(x.tolist()).reshape(32, 32, 3)
                    ax.imshow(img)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('auto')

            plt.savefig(style_dir / ('epoch_%d.png' % epoch))
            plt.close()

    save_models(decoder, encoder, discriminator, epoch)
    print(UNIQUE_RUN_ID)
