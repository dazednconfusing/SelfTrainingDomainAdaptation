"""
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Build the encoder
"""


def encoder(latent_dim, input_shape):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


"""
## Build the decoder
"""


def decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same"
    )(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


"""
## Classifier
"""


def classifier(num_labels, latent_dim):
    latent_inputs_class = keras.Input(shape=(latent_dim,))
    class_outputs = layers.Dense(num_labels, activation=tf.nn.softmax)(
        latent_inputs_class
    )
    return keras.Model(latent_inputs_class, class_outputs, name="classifier")


"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(keras.Model):
    def __init__(
        self, num_labels, input_shape=(28, 28, 1), latent_dim=20, cw=20, rw=10, **kwargs
    ):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder(latent_dim, input_shape)
        self.decoder = decoder(latent_dim)
        self.classifier = classifier(num_labels, latent_dim)
        self.latent_dim = latent_dim
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.classifier_loss_tracker = keras.metrics.Mean(name="classifier_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

        self.classifier_weight = cw
        self.recon_weight = rw

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.classifier_loss_tracker,
            self.kl_loss_tracker,
            self.accuracy_tracker,
        ]

    def train_step(self, data):
        x_, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x_)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x_, reconstruction), axis=(1, 2)
                )
            )
            outputs = self.classifier(z)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            classifier_loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(y, outputs)
            )
            total_loss = (
                self.recon_weight * reconstruction_loss
                + kl_loss
                + self.classifier_weight * classifier_loss
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.classifier_loss_tracker.update_state(classifier_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.accuracy_tracker.update_state(y, outputs)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "classifier_loss": self.classifier_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "sparse_categorical_accuracy": self.accuracy_tracker.result(),
        }

    def test_step(self, data):
        x_, y = data
        z_mean, z_log_var, z = self.encoder(x_)
        reconstruction = self.decoder(z)
        reconstruction_loss = (
            tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x_, reconstruction), axis=(1, 2)
                )
            )
            / 784
        )
        outputs = self.classifier(z)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) / self.latent_dim
        classifier_loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(y, outputs)
        )
        total_loss = (
            self.recon_weight * reconstruction_loss
            + kl_loss
            + self.classifier_weight * classifier_loss
        )
        self.total_loss_tracker.update_state(total_loss)
        self.accuracy_tracker.update_state(y, outputs)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.classifier_loss_tracker.update_state(classifier_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "classifier_loss": self.classifier_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "sparse_categorical_accuracy": self.accuracy_tracker.result(),
        }

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        outputs = self.classifier(z)
        return outputs


"""
## Train the VAE
"""

# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# mnist_digits = np.concatenate([x_train, x_test], axis=0)
# mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

# vae = VAE(encoder, decoder)
# vae.compile(optimizer=keras.optimizers.Adam())
# vae.fit(mnist_digits, epochs=30, batch_size=128)
