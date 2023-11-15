import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from pathlib import Path
import cv2
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.experimental.list_physical_devices("GPU")
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)
#         exit(-1)
PROJECT_ROOT = Path.cwd()
UNIQUE_RUN_ID = 'MAAElayer_celeba_3d_lambda_15z'
generator = tf.keras.models.load_model(f'save_models/{UNIQUE_RUN_ID}/decoder_200.model')
z_dim = 15
img_size = 128
num_c = 3


def generate_fake_images(model, NUM, batch_size=100):
    """ Generate subplots with generated examples. """
    fake_images = []
    num_batches = NUM // batch_size
    for i in range(num_batches):
        z = tf.random.normal([batch_size, z_dim], mean=0.0, stddev=5.0)
        x = model(z, training=False)
        x = tf.reshape(x, (-1, img_size, img_size, num_c))
        x = 2. * x - 1
        new_size = (299, 299)  # 新的图像尺寸
        img = tf.image.resize(x, new_size, method='bilinear')
        fake_images.append(img)
    fake_images = np.concatenate(fake_images, axis=0)
    return fake_images


# 计算图像的激活统计信息（平均值和协方差矩阵）
def calculate_activation_statistics(images, model):
    act_values = model.predict(images)
    act_values = act_values.reshape(act_values.shape[0], -1)
    mean = np.mean(act_values, axis=0)
    cov = np.cov(act_values, rowvar=False)
    return mean, cov


# 计算两个分布之间的 Fréchet 距离
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    sqrt_cov1, sqrt_cov2 = sqrtm(sigma1), sqrtm(sigma2)
    cov_mean = sqrtm(np.dot(sqrt_cov1, np.dot(sigma2, sqrt_cov1)))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid_score = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * cov_mean)
    return fid_score


def calculate_fid(real_images, fake_images, batch_size=100):
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    real_images = preprocess_input(real_images)
    fake_images = preprocess_input(fake_images)
    real_mean, real_cov = None, None
    num_batches = len(real_images) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        real_images_batch = real_images[start:end]
        batch_mean, batch_cov = calculate_activation_statistics(real_images_batch, model)
        if real_mean is None or real_cov is None:
            real_mean, real_cov = batch_mean, batch_cov
        else:
            real_mean = (real_mean * (i) + batch_mean) / (i + 1)
            real_cov = (real_cov * (i) + batch_cov) / (i + 1)
    fake_mean, fake_cov = calculate_activation_statistics(fake_images, model)
    fid_score = calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
    return fid_score


real_image_paths = glob.glob(str(PROJECT_ROOT /'celeba'/ 'celeba_data' / 'test_data' / '*.*'))  # 获取目录下所有文件的文件路径列表
real_images = []
for image_path in real_image_paths[:10000]:  # 只使用前10000张真实图像进行计算
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    real_images.append(img)
real_images = np.array(real_images)
real_images = real_images.astype('float32') / 127.5 - 1
NUM = 10000  # 生成假图像的数量
batch_size = 100  # 每批生成的假图像数量
fake_images = generate_fake_images(generator, NUM, batch_size=batch_size)

fid_score = calculate_fid(real_images, fake_images, batch_size=100)
print("FID Score:", fid_score)
print(UNIQUE_RUN_ID)
