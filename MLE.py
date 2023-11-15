"""
Deterministic supervised adversarial autoencoder.

 We are using:
    - Gaussian distribution as prior distribution.
    - dense layers.
    - Cyclic learning rate
"""
import tensorflow as tf
import gc
import os
import time
import numpy as np
from pathlib import Path
import random
from mindspore import context
import mindspore.numpy
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt
import re
from utils import DataPreparer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
PROJECT_ROOT = Path.cwd()
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        exit(-1)
# -------------------------------------------------------------------------------------------------------------
random_seed = 42
random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed)  # set random seed for numpy
tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu

UNIQUE_RUN_ID = 'MAAElayer_cifar10_3d_lambda_15z'

# 使用正则表达式提取信息
match = re.match(r'^(\w+)_(\w+)_(\d+)d_(\w+)_(\d+)z$', UNIQUE_RUN_ID)

# 提取信息
d = int(match.group(3))
dataset = match.group(2)
style = match.group(4)
z_dim = int(match.group(5))

# 打印结果
print(f"d: {d}")
print(f"dataset: {dataset}")
print(f"style: {style}")
print(f"z: {z_dim}")

# -------------------------------------------------------------------------------------------------------------
if dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    img_size = 28
    num_c = 1
    test_size = 1 / 60
    epoch = 4
else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    img_size = 32
    num_c = 3
    test_size = 1 / 50
    epoch = 50
BUFFER_SIZE = 60000
BATCH_SIZE = 100
sigma = None

# squeeze1 = mindspore.ops.Squeeze(0.1)
# -------------------------------------------------------------------------------------------------------------
# Loading data
# -------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------
# Create the dataset iterator

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train,  # 把上面剩余的 x_train, y_train继续拿来切
    test_size=test_size  # test_size默认是0.25
)
# x_train = (x_train.astype('float32')-127.5) / 127.5
x_valid = x_valid.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape(x_train.shape[0], img_size , img_size , num_c)
x_valid = x_valid.reshape(x_valid.shape[0], img_size * img_size * num_c)
x_test = x_test.reshape(x_test.shape[0], img_size * img_size * num_c)
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# UNIQUE_RUN_ID = 'supervised_aae_mixture_posterior_dense_2z_testlr'
def gaussian_mixture(batch_size, labels, n_classes):
    x_stddev = 5.
    y_stddev = 1.
    shift = 10.

    x = np.random.normal(0, x_stddev, batch_size).astype("float32") + shift

    y = np.random.normal(0, y_stddev, batch_size).astype("float32")
    z = np.array([[xx, yy] for xx, yy in zip(x, y)])

    def rotate(z, label):
        angle = label * 2.0 * np.pi / n_classes
        rotation_matrix = np.array(
            [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        )
        z[np.where(labels == label)] = np.array(
            [
                rotation_matrix.dot(np.array(point))
                for point in z[np.where(labels == label)]
            ]
        )
        return z

    for label in set(labels):
        rotate(z, label)

    return z


z = tf.random.normal([x_test.shape[0], z_dim], mean=0.0, stddev=5.)
# +++++++++++++++++++supervised
# n_labels = 10
# label_sample = np.random.randint(0, n_labels, size=[x_test.shape[0]])
# print(label_sample)
# z = gaussian_mixture(x_test.shape[0], label_sample, n_labels)
decoder = tf.keras.models.load_model("save_models/" + UNIQUE_RUN_ID + f"/decoder_{epoch}.model")
discriminator = tf.keras.models.load_model("save_models/" + UNIQUE_RUN_ID + f"/discriminator2_{epoch}.model")
discriminator.summary()
# max:3d->4d->5d
# generator = tf.keras.models.load_model("./gan_75z/generator_24.model/")

def log_mean_exp(a):
    max_ = a.max(axis=1)
    max2 = np.reshape(max_, (max_.shape[0], 1))
    return max_ + np.log(np.exp(a - max2).mean(1))


def mind_parzen(x, mu, sigma):
    a = (np.reshape(x, (x.shape[0], 1, x.shape[-1])) - np.reshape(mu, (1, mu.shape[0], mu.shape[-1]))) / sigma
    a5 = -0.5 * (a ** 2).sum(2)
    E = log_mean_exp(a5)
    t4 = sigma * np.sqrt(np.pi * 2)
    t5 = np.log(t4)
    Z = mu.shape[1] * t5
    return E - Z


def get_nll(x, samples, sigma, batch_size):
    '''get_nll'''
    inds = range(x.shape[0])
    inds = list(inds)
    n_batches = int(np.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = Tensor(np.array([]).astype(np.float32))
    for i in range(n_batches):
        begin = time.time()
        nll = mind_parzen(x[inds[i::n_batches]], samples, sigma)
        end = time.time()
        times.append(end - begin)
        nlls = tf.concat([nlls, nll], 0)

        if i % 10 == 0:
            print(i, np.mean(times), np.mean(nlls))

    return nlls


def cross_validate_sigma(samples, data, sigmas, batch_size):
    '''cross_validate_sigma'''
    lls = Tensor(np.array([]).astype(np.float32))
    for sigma in sigmas:
        print(sigma)
        tmp = get_nll(data, samples, sigma, batch_size=batch_size)
        tmp = np.mean(tmp)
        tmp = np.reshape(tmp, (1, 1))
        tmp = tf.squeeze(tmp, 1)
        lls = tf.concat([lls, tmp], 0)
        gc.collect()

    ind = np.argmax(lls)
    return sigmas[ind]


gen = decoder(z, training=False)
# gen=(gen*127.5+125.5)/255.
# gen = gen.astype('float32')

gen = np.reshape(gen, (x_test.shape[0], img_size * img_size * num_c))
print('gen_image shape:', gen.shape)
# cross validate sigma
if sigma is None:
    sigma_range = np.logspace(start=-1, stop=-0.3, num=20)
    # sigma_range = np.linspace(start=0.1, stop=0.5, num=20)
    sigma = cross_validate_sigma(
        gen, x_valid, sigma_range, batch_size=BATCH_SIZE
    )
    sigma = sigma
else:
    sigma = float(sigma)
print("Using Sigma: {}".format(sigma))


# fit and evaulate
# gen_imgs

def parzen(gen):
    '''parzen'''

    gc.collect()

    ll = get_nll(x_test, gen, sigma, batch_size=BATCH_SIZE)
    se = np.std(ll) / np.sqrt(x_test.shape[0])

    print("Log-Likelihood of test set = {}, se: {}".format(np.mean(ll), se))
    print(UNIQUE_RUN_ID)

    return np.mean(ll), se


mean_ll, se_ll = parzen(gen)
