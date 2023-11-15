import numpy as np
from sklearn.datasets import make_swiss_roll
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import sin, cos, sqrt

def plot_semi_sample(decoder, sample_dir, img_shape, epoch=0):
    nx, ny = 10, 10

    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
    i = 0
    for t in range(nx):
        for r in range(ny):
            label = np.random.randint(t, t + 1, size=[1])
            # label_sample_one_hot = tf.one_hot(label, n_labels)
            real_distribution = gaussian_mixture(1, label, n_labels)
            # dec_input = tf.concat(
            #     [real_distribution,label_sample_one_hot], axis=1
            # )
            x = decoder(real_distribution, training=False).numpy()
            ax = plt.subplot(gs[i])
            i += 1
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

    plt.savefig(style_dir / ('epoch_%d.png' % epoch))
    plt.close()


class DataPreparer:
    def __init__(self, num_samples=100, **kwargs):
        self.kwargs = kwargs
        self.mean = 0
        self.std = self.kwargs.get('std')
        self.num_samples = num_samples
        self.prior_type = self.kwargs.get('prior_type')
        self.latent_dim = self.kwargs.get('latent_dim')

    def sample_normal(self):
        # Sample from a normal distribution
        z = np.random.normal(mean, self.std, size=(self.num_samples, self.latent_dim))
        return z.astype('float32')

    def sample_mixture_of_normals(self, batch_size, labels, n_classes=10):
        x_stddev = self.std
        y_stddev = self.std/5
        shift = self.std*2

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

    def sample_swiss_roll(self):
        # Sample from the Swiss Roll dataset
        z, _ = make_swiss_roll(n_samples=self.batch_size, noise=0.1)
        z = z[:, [0, 2]]
        z /= 10.0  # scale down the data to a smaller range
        return z.astype('float32')

    def plot(self):
        if self.prior_type == 'normal_gaussian':
            z = self.sample_normal()
        elif self.prior_type == 'mixture_gaussian':
            z = self.sample_mixture_of_normals(num_components, stds, weights)
        elif self.prior_type == 'swiss_roll':
            z = self.sample_swiss_roll()
        else:
            raise ValueError('Invalid prior type')
        return z

    def save_models(self, UNIQUE_RUN_ID, models, epoch):
        """ Save models at specific point in time. """
        encoder, decoder, discriminator = models
        tf.keras.models.save_model(
            decoder,
            f'./save_models/{UNIQUE_RUN_ID}/decoder_{epoch}.model',
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )
        tf.keras.models.save_model(
            encoder,
            f'./save_models/{UNIQUE_RUN_ID}/encoder_{epoch}.model',
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )
        for ind in range(len(discriminator)):
            tf.keras.models.save_model(
                discriminator[ind],
                f'./save_models/{UNIQUE_RUN_ID}/discriminator{ind}_{epoch}.model',
                overwrite=True,
                include_optimizer=True,
                save_format=None,
                signatures=None,
                options=None
            )

    def plot_latent(self, encoder, x_test, y_test, latent_space_dir, epoch=0):
        # Latent Space
        z, z_std = encoder(x_test, training=False)
        label_onehot = tf.one_hot(y_test, 10)
        labels = [np.argmax(one_hot) for one_hot in label_onehot]
        label_list = list(labels)
        # label_list = list(y_test)
        fig = plt.figure()
        classes = set(list(label_list))
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        kwargs = {'alpha': 0.8, 'c': [colormap[i] for i in label_list]}
        ax = plt.subplot(111, aspect='equal')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles = [mpatches.Circle((0, 0), label=class_, color=colormap[i])
                   for i, class_ in enumerate(classes)]
        ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                  fancybox=True, loc='center left')
        plt.scatter(z[:, 0], z[:, 1], s=2, **kwargs)
        ax.set_xlim([-self.std * 4, self.std * 4])
        ax.set_ylim([-self.std * 4, self.std * 4])

        plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
        plt.close('all')

    def plot_recon(self, models, x_test, reconstruction_dir, img_shape, epoch=0):
        img_size, num_c = img_shape
        encoder, decoder, _ = models
        n_digits = 25
        # Reconstruction
        z, z_std = encoder(x_test[:n_digits], training=False)
        # z = reparameterization(z_mean, z_std)

        x_test_decoded = decoder(z, training=False)
        x_test_decoded = np.reshape(x_test_decoded, [-1, img_size, img_size, num_c])
        fig = plt.figure(figsize=(5, 5))
        for i in range(n_digits):
            # Plot
            plt.subplot(5, 5, i + 1)
            if num_c == 3:
                plt.imshow(x_test_decoded[i])
            else:
                plt.imshow(x_test_decoded[i][:, :, 0], cmap='gray')
            plt.axis('off')
        plt.savefig(reconstruction_dir / ('epoch_%d.png' % epoch))

    def plot_sample(self, decoder, sample_dir, img_shape, epoch=0):
        # Sampling
        num = 5
        img_size, num_c = img_shape
        """ Generate subplots with generated examples. """
        z = tf.random.normal([num * num, self.latent_dim], mean=0.0, stddev=self.std)
        images = decoder(z, training=False)
        plt.figure(figsize=(num, num),dpi=300)
        for i in range(num * num):
            # Get image and reshape
            image = images[i]
            image = np.reshape(image, [img_size, img_size, num_c])
            # Plot
            plt.subplot(num, num, i + 1)
            if num_c == 1:
                plt.imshow(image[:, :, 0], cmap='gray')
            else:
                plt.imshow(image[:, :, :])
            plt.axis('off')
        plt.savefig(sample_dir / ('epoch_%d.png' % epoch))

