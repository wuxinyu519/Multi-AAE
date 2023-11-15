import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
import os

# Load MNIST dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
UNIQUE_RUN_ID = 'MAAE_mnist_3d_lambda_2z'
encoder = tf.keras.models.load_model("save_models/" + UNIQUE_RUN_ID + "/encoder_50.model")
std = 5.
x_test = x_test[:1000].reshape(x_test[:1000].shape[0], 28 * 28 * 1)
x_test = x_test.astype('float32') / 255.
x_test_kl=x_test
x_test, z_std = encoder(x_test, training=False)
x_test_kl, _ = encoder(x_test_kl, training=False)
mu = x_test_kl
# sigma = tf.exp(0.5 * z_std)
# print(sigma)
# print(x_test.shape)
x_test = np.array(x_test)
y_test = y_test[:1000]
# 将一部分样本标记为无标签
num_labeled_samples = 100
labeled_indices = np.arange(num_labeled_samples)
unlabeled_indices = np.arange(num_labeled_samples, len(x_test))

# 初始化SVM模型
svm = SVC(kernel='linear', probability=True)
svm.fit(x_test[labeled_indices], y_test[labeled_indices])

# 迭代停止条件
max_iterations = 20
min_confidence_ratio = 0.2

# 进行多次迭代
for iteration in range(max_iterations):
    # 1. 对未标记样本进行预测，得到置信度
    if len(unlabeled_indices) > 0:
        # 1. 对未标记样本进行预测，得到置信度矩阵
        confidences = svm.decision_function(x_test[unlabeled_indices])  # 置信度矩阵，形状为 (900, 10)
        confidences = tf.reduce_max(tf.nn.softmax(confidences), axis=-1)
        # 根据概率值选择高置信度样本
        sorted_indices = np.argsort(confidences)[::-1]
        print(len(sorted_indices))
        num_high_confidence_samples = int(len(unlabeled_indices) * min_confidence_ratio)
        high_confidence_indices = sorted_indices[:num_high_confidence_samples]

        # 如果没有高置信度样本
        if len(high_confidence_indices) == 0:
            print("未找到置信度大于等于 min_confidence 的样本。")
            break  # 或者采取适当的操作
        else:
            # 将高置信度样本的索引加入到标记样本的索引中
            labeled_indices = np.concatenate([labeled_indices, unlabeled_indices[high_confidence_indices]])
        # 4. 重新训练 SVM 模型
        svm.fit(x_test[labeled_indices], y_test[labeled_indices])

        # 更新未标记样本的索引
        unlabeled_indices = np.setdiff1d(np.arange(len(x_test)), labeled_indices)

        # 输出当前迭代的损失
        loss = 1 - svm.score(x_test[num_labeled_samples:], y_test[num_labeled_samples:])
        print("Iteration %d, Loss: %f" % (iteration + 1, loss))


def calculate_gaussian_kl_divergence(m1, m2, v1, v2):
    # print(v1)
    # v1 = tf.maximum(v1, 1e-8)
    return tf.reduce_mean(
        tf.math.log(v2 / v1) + tf.divide(tf.add(tf.square(v1), tf.square(m1 - m2)), 2 * tf.square(v2)) - 0.5)


def kl_divergence(mu, logvar):
    var = tf.exp(logvar)
    kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - var, axis=-1)
    return kl


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


# print(z_std)
# print(sigma)
# v1 = tf.maximum(sigma, 1e-8)
# print(v1)
# kl_loss = compute_kl_div(mu, sigma)
# kl_loss = kl_divergence(mu,z_std)

real_distribution = tf.random.normal([10000, x_test.shape[1]], mean=0.0, stddev=std)
kl_loss = compute_mmd(real_distribution, mu)

print(f'kl_loss is {tf.reduce_mean(kl_loss)}')
print(f'result of {UNIQUE_RUN_ID} with {x_test.shape[1]} dim')
