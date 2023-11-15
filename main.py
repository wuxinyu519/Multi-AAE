import os
import random
from pathlib import Path
import argparse
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from MAAE_model import MAAE
from utils import DataPreparer


def load_data(**kwargs):
    """
    Load the specified dataset and split it into training, validation, and testing sets.
    :param dataset_name: Name of the dataset to load ("mnist", "cifar10", or "celeba")
    :return: Tuple containing (x_train, y_train), (x_valid, y_valid), and (x_test, y_test) data splits
    """
    dataset_name = kwargs.get("dataset")
    img_size = kwargs.get("img_size")
    num_c = kwargs.get("num_c")
    batch_size = kwargs.get("batch_size")
    print(f'load {dataset_name}... ')
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # x_train=x_train[:500]
        # y_train = y_train[:500]
        # Split the training set into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1 / 60)
        # Create the dataset iterator
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape(x_train.shape[0], img_size * img_size * num_c)
        x_test = x_test.reshape(x_test.shape[0], img_size * img_size * num_c)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train))
        train_dataset = train_dataset.shuffle(buffer_size=60000)
        train_dataset = train_dataset.batch(batch_size)
    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # Split the training set into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1 / 50)
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        x_train = x_train.reshape(x_train.shape[0], img_size, img_size, num_c)
        x_test = x_test.reshape(x_test.shape[0], img_size, img_size, num_c)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train))
        train_dataset = train_dataset.shuffle(buffer_size=60000)
        train_dataset = train_dataset.batch(batch_size)
    elif dataset_name == "celeba":
        # Code to load CelebA dataset goes here
        # For example:
        PROJECT_ROOT = Path.cwd()
        train_data = PROJECT_ROOT / 'celeba'/ 'celeba_data' / 'train_data'
        test_data = PROJECT_ROOT /'celeba' / 'celeba_data' / 'test_data'

        x_train = tf.keras.preprocessing.image_dataset_from_directory(
            train_data,
            label_mode=None,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
        )

        x_train = x_train.map(lambda x: x / 255.0)
        train_dataset = x_train.prefetch(buffer_size=10 * batch_size)
        x_test = tf.keras.preprocessing.image_dataset_from_directory(
            test_data,
            label_mode=None,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=False
        )
        x_test = x_test.map(lambda x: x / 255.0)
        y_test= None
        x_valid = None
        y_valid = None
    else:
        raise ValueError("Invalid dataset name. Must be one of 'mnist', 'cifar10', or 'celeba'.")

    return train_dataset, (x_valid, y_valid), (x_test, y_test)


def reparameterization(z_mean, z_std, std):
    # if len(z_mean_std)==0:
    #     # Probabilistic with Gaussian posterior distribution
    #     epsilon = tf.random.normal(shape=z_mean.shape, mean=0., stddev=std)
    #     z = z_mean + tf.exp(0.5 * z_std) * epsilon
    # else:
    # Probabilistic with Gaussian posterior distribution
    epsilon = tf.random.normal(shape=z_mean.shape, mean=0., stddev=std)
    z = z_mean + tf.exp(0.5 * z_std) * epsilon
    return z


def autoencoder_loss(inputs, reconstruction, mse):
    return mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output, NUM_OF_D, cross_entropy):
    # d_loss = [tf.reduce_mean(-tf.math.log(real_output[ind]) - tf.math.log(1 - fake_output[ind])) for ind in
    #           range(NUM_OF_D)]
    loss_real = [cross_entropy(tf.ones_like(real_output[ind]), real_output[ind]) for ind in range(NUM_OF_D)]
    loss_fake = [cross_entropy(tf.zeros_like(fake_output[ind]), fake_output[ind]) for ind in range(NUM_OF_D)]
    d_loss = [loss_real[i] + loss_fake[i] for i in range(NUM_OF_D)]
    return d_loss
    # return tf.reduce_mean(-tf.math.log(real_output) - tf.math.log(1 - fake_output))


def mix_pre(G_losses, lam):
    used_l = tf.nn.softplus(lam)
    weights = tf.exp(used_l * G_losses)
    denom = tf.reduce_sum(weights)
    weights = tf.math.divide(weights, denom)
    print('lam_weights shape', weights.shape)
    g_loss = tf.reduce_sum(weights * G_losses)
    return g_loss, used_l


def generator_loss(fake_output, NUM_OF_D, style, cross_entropy, lam=None):
    G_losses = [cross_entropy(tf.ones_like(fake_output[ind]), fake_output[ind]) for ind in range(NUM_OF_D)]
    # G_losses = [-tf.reduce_mean(tf.math.log(fake_output[ind])) for ind in range(NUM_OF_D)]
    if style == 'lambda':
        gen_loss, used_l = mix_pre(G_losses, lam)
        g_loss = gen_loss - 0.001 * used_l
        return g_loss, used_l, lam
    elif style == 'mean':
        used_l=0
        return tf.reduce_mean(G_losses),used_l
    else:
        used_l = 0
        return tf.reduce_max(G_losses),used_l

    #


@tf.function
def train_step(model, batch_x, lam, optimizers, std, NUM_OF_D, style, losses, z_dim):
    mse, cross_entropy, accuracy = losses
    ae_optimizer, dc_optimizer, gen_optimizer = optimizers
    with tf.GradientTape(persistent=True) as ae_tape:
        z = model.encode(batch_x, training=True)
        decoder_output = model.decode(z, training=True)
        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, mse)

    ae_grads = ae_tape.gradient(ae_loss, model.encoder.trainable_variables + model.decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, model.encoder.trainable_variables + model.decoder.trainable_variables))

    # Discriminator
    with tf.GradientTape(persistent=True) as tape:
        real_distribution = [tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=std) for _ in range(NUM_OF_D)]

        z = model.encode(batch_x, training=True)
        dc_real = [model.discriminator[ind](real_distribution[ind], training=True) for ind in
                   range(NUM_OF_D)]
        dc_fake = [model.discriminator[ind](z, training=True) for ind in
                   range(NUM_OF_D)]
        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, NUM_OF_D, cross_entropy)
        # Discriminator Acc
        dc_acc = [accuracy(tf.concat([tf.ones_like(dc_real[ind]), tf.zeros_like(dc_fake[ind])], axis=0),
                           tf.concat([dc_real[ind], dc_fake[ind]], axis=0)) for ind in range(NUM_OF_D)]

        # Generator loss
        if style == 'lambda' and NUM_OF_D > 1:
            gen_loss, used_l, lam = generator_loss(dc_fake, NUM_OF_D, style, cross_entropy, lam)
            gen_vars = model.encoder.trainable_variables + lam
        else:
            gen_loss, used_l = generator_loss(dc_fake, NUM_OF_D, style, cross_entropy)
            gen_vars = model.encoder.trainable_variables

    dc_grads_discriminator = [tape.gradient(dc_loss[ind], model.discriminator[ind].trainable_variables) for ind in
                              range(NUM_OF_D)]
    dc_grads_generator = tape.gradient(gen_loss, gen_vars)

    for ind in range(NUM_OF_D):
        dc_optimizer.apply_gradients(zip(dc_grads_discriminator[ind], model.discriminator[ind].trainable_variables))

    gen_optimizer.apply_gradients(zip(dc_grads_generator, gen_vars))

    del tape, ae_tape

    return ae_loss, dc_loss, dc_acc, gen_loss, used_l


def train(args):
    # -------------------------------------------------------------------------------------------------------------
    random_seed = 0
    random.seed(random_seed)  # set random seed for python
    np.random.seed(random_seed)  # set random seed for numpy
    tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)

    # -------------------------------------------------------------------------------------------------------------
    # Set up output directory
    UNIQUE_RUN_ID = f'MAAE_{args.dataset}_{args.num_of_d}d_{args.style}_{args.latent_dim}z'
    PROJECT_ROOT = Path.cwd()
    output_dir = Path(args.output_dir)
    experiment_dir = output_dir / f'MAAE_{args.dataset}_mle_{args.num_of_d}d_{args.style}_{args.latent_dim}z'
    experiment_dir.mkdir(parents=True, exist_ok=True)
    latent_space_dir = experiment_dir / 'latent_space'
    latent_space_dir.mkdir(exist_ok=True)
    reconstruction_dir = experiment_dir / 'reconstruction'
    reconstruction_dir.mkdir(exist_ok=True)
    sample_dir = experiment_dir / 'sample'
    sample_dir.mkdir(exist_ok=True)
    lam = [tf.Variable(tf.constant(args.lam))]
    # Load data
    train_dataset, (x_valid, y_valid), (x_test, y_test) = load_data(dataset=args.dataset,
                                                                    img_size=args.img_size,
                                                                    num_c=args.num_c,
                                                                    batch_size=args.batch_size)

    # Create MAAE model
    maae = MAAE(num_of_d=args.num_of_d,
                latent_dim=args.latent_dim,
                img_size=args.img_size,
                num_c=args.num_c,
                std=args.std,
                training_style='unsupervised'
                )
    maae.encoder.summary()
    maae.decoder.summary()
    # maae.discriminator.summary()
    util = DataPreparer(
        num_samples=args.batch_size,
        std=args.std,
        noise_tyle=args.prior_style,
        latent_dim=args.latent_dim,
    )

    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=args.ae_lr)
    dc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.dc_lr)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=args.gen_lr)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()
    accuracy = tf.keras.metrics.BinaryAccuracy()
    # Training loop
    if not os.path.exists(f'./cruves/{UNIQUE_RUN_ID}'):
        os.mkdir(f'./cruves/{UNIQUE_RUN_ID}')
    writer = tf.summary.create_file_writer(f'./cruves/{UNIQUE_RUN_ID}/')
    with writer.as_default():
        enc_std = []
        enc_dec_std = []
        all_std = []
        global_step = 0

        for epoch in range(args.n_epochs):

            start = time.time()
            # Train model

            for batch, (batch_x) in enumerate(train_dataset):
                global_step = global_step + 1

                ae_loss, dc_loss, dc_acc, gen_loss, used_l = train_step(maae,
                                                                        batch_x,
                                                                        lam,
                                                                        optimizers=[ae_optimizer, dc_optimizer,
                                                                                    gen_optimizer],
                                                                        std=args.std,
                                                                        NUM_OF_D=args.num_of_d,
                                                                        style=args.style,
                                                                        losses=[mse, cross_entropy, accuracy],
                                                                        z_dim=args.latent_dim)

                enc_std.append(np.mean(gen_loss))
                enc_dec_std.append(np.mean(ae_loss))
                all_std.append(np.mean(gen_loss) + np.mean(ae_loss))
                if (batch + 1) % 500 == 0:
                    tf.summary.scalar("enc_std", np.mean(np.std(enc_std)), global_step)
                    tf.summary.scalar("enc_dec_std", np.mean(np.std(enc_dec_std)), global_step)
                    tf.summary.scalar("all_std", np.mean(np.std(all_std)), global_step)
                    enc_std.clear()
                    enc_dec_std.clear()
                    all_std.clear()
                if global_step % 10 == 0:
                    tf.summary.scalar("ae_loss", np.mean(ae_loss), global_step)
                    tf.summary.scalar("dc_loss", np.mean(dc_loss), global_step)
                    tf.summary.scalar("gen_loss", np.mean(gen_loss), global_step)
                    tf.summary.scalar("dc_acc", np.mean(dc_acc), global_step)
                    tf.summary.scalar("used_l", np.mean(used_l), global_step)
                # if global_step % 100 == 0:
                #     util.plot_sample(maae.decoder, sample_dir, img_shape=(args.img_size, args.num_c), epoch=global_step)

            epoch_time = time.time() - start
            print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
                  .format(epoch, epoch_time,
                          epoch_time * (args.n_epochs - epoch),
                          np.mean(ae_loss),
                          np.mean(dc_loss),
                          np.mean(dc_acc),
                          np.mean(gen_loss)

                          ))

            if epoch % 5 == 0:
                models = (maae.encoder, maae.decoder, maae.discriminator)
                util.save_models(UNIQUE_RUN_ID, models, epoch=epoch)
                if args.dataset != 'celeba':
                    util.plot_latent(maae.encoder,x_test, y_test, latent_space_dir, epoch=epoch)
                    util.plot_recon(models, x_test, reconstruction_dir, img_shape=(args.img_size,args.num_c), epoch=epoch)
            util.plot_sample(maae.decoder, sample_dir, img_shape=(args.img_size, args.num_c), epoch=epoch)
            # util.plot_sample(maae.decoder, sample_dir, img_shape=(args.img_size, args.num_c), epoch=global_step)
    util.save_models(UNIQUE_RUN_ID, models, epoch)

    print(f'finished: {UNIQUE_RUN_ID}')


if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------------------------
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='MAAE')
    parser.add_argument('--gpu', type=int, default=0, help='Index of the GPU to use')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'celeba'],
                        help='Dataset to use (mnist, cifar10, or celeba)')
    parser.add_argument('--latent_dim', type=int, default=8, help='Dimension of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='shape of image')
    parser.add_argument('--num_c', type=int, default=1, help='number of channel')
    parser.add_argument('--num_of_d', type=int, default=3, help='Number of discriminators')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=201, help='Number of epochs to train for-cifar:101:mnist:201,celeba:51')
    parser.add_argument('--ae_lr', type=float, default=0.0002, help='Learning rate for autoencoder')
    parser.add_argument('--gen_lr', type=float, default=0.0001, help='Learning rate for generator')
    parser.add_argument('--dc_lr', type=float, default=0.0001, help='Learning rate for discriminator')
    parser.add_argument('--lam', type=float, default=0.01, help='soft-ensemble parameter')
    parser.add_argument('--std', type=float, default=5, help='standard deviation of prior')
    parser.add_argument('--style', type=str, default='lambda',
                        choices=['lambda', 'mean', 'max'],
                        help='ensemble style to use ')
    parser.add_argument('--prior_style', type=str, default='normal_gaussian',
                        choices=['normal_gaussian', 'mixture_gaussian', 'swiss_roll'],
                        help='prior style ')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    args = parser.parse_args()
    train(args)
