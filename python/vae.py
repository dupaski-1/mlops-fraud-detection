import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd

tfk = tf.keras
tfpl = tfp.layers

# TODO: Inherit from the appropriate Keras Layer class
class Encoder(layers.Layer):
    def __init__(self, input_shape=(28,), units=50, activation='tanh', latent_dim=10, name='enc', distribution=tfd.MultivariateNormalDiag, **kwargs):
        # TODO: Call the parent class constructor. Arnaud: super to refer to the superclass.
        # name is a variable not directly used by Encoder class, meaning it must be used higher up.
        # kwargs are by definition non essential arguments of that class but coming from higher up.
        super().__init__(name=name, **kwargs)

        # We provide the hidden layer architecture for you:
        self.hidden_layers = Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(units, activation=activation),
            layers.Dense(units, activation=activation),
            layers.Dense(2 * latent_dim),
        ])

        # TODO: Store the latent_dim as an instance attribute
        # Arnaud: the "_" at the beginning of the attribute is a signal to other programmers
        # that this is a variable meant to not be touched by the user. 
        # This code structure is not necessary but in this lab it was chosen by the teacher.
        # Structuring that way intends to make the attribute a read-only and accessible solely through
        # the function "latent_dim", which we define below and only returns, but never edits the variable 
        self._latent_dim = latent_dim

        # TODO: Define the DistributionLambda layer (qz_x)
        # Hint: use tfpl.DistributionLambda and the passed 'distribution'
        # Arnaud: the teacher forgot to delete that line.. so it is here already..
        self.qz_x = tfpl.DistributionLambda(lambda t: distribution(t[..., :latent_dim], tf.math.exp(t[..., latent_dim:])))

    @property
    def latent_dim(self):
        # TODO: Return the stored latent dimension
        return self._latent_dim

    def call(self, inputs):
        # TODO: Pass the inputs through the hidden layers
        # TODO: Pass the output of the hidden layers through qz_x
        # TODO: Return the resulting distribution
        return self.qz_x(self.hidden_layers(inputs))


# TODO: Inherit from the appropriate Keras Layer class
class Decoder(layers.Layer):
    def __init__(self, latent_dim=10, units=50, activation='tanh', output_dim=28, name='dec', distribution=tfd.MultivariateNormalDiag, **kwargs):
        # TODO: Call the parent class constructor
        super().__init__(name=name,**kwargs)

        self.hidden_layers = Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(units, activation=activation),
            layers.Dense(units, activation=activation),
        ])

        # TODO: Define a Dense layer for the mean (mu) with 'sigmoid' activation
        self.mu = layers.Dense(output_dim,activation='sigmoid')
        # TODO: Define a Dense layer for the log variance (log_sigma)
        self.log_sigma = layers.Dense(output_dim)
        # TODO: Store the latent_dim as an instance attribute
        self._latent_dim = latent_dim

    @property
    def latent_dim(self):
        # TODO: Return the stored latent dimension
        return self._latent_dim

    def call(self, inputs):
        # TODO: Pass the inputs through the hidden_layers
        lay = self.hidden_layers(inputs)
        # TODO: Get mu and log_sigma from their respective layers
        mu = self.mu(lay)
        log_sigma = self.log_sigma(lay)

        # TODO: Create and return the distribution
        # Hint: tfd.MultivariateNormalDiag(mu, tf.math.exp(log_sigma))
        return tfd.MultivariateNormalDiag(mu, tf.math.exp(log_sigma))

    # TODO: Inherit from the appropriate Keras Model class
class VAE(tfk.Model):
    def __init__(self, encoder, decoder, optimizer, name='VAEprob', **kwargs):
        # TODO: Call the parent class constructor
        super().__init__(name=name,**kwargs)

        # TODO: Store encoder, decoder, and optimizer as instance attributes
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer

        # TODO: Extract and store latent_dim from the encoder
        self._latent_dim = encoder.latent_dim

        # TODO: Define the prior distribution (pz)
        self.pz = tfd.MultivariateNormalDiag(loc=tf.zeros(self._latent_dim), scale_diag=tf.ones(self._latent_dim))
        # Hint: tfd.MultivariateNormalDiag with loc=0 and scale_diag=1

    def call(self, inputs):
        # TODO: Get the posterior distribution (qz_x) from the encoder
        qz_x = self.encoder(inputs)
        # TODO: Sample a latent vector (z) from qz_x
        z = qz_x.sample()
        # TODO: Get the reconstruction distribution (px_z) from the decoder using z
        px_z = self.decoder(z)
        # TODO: Calculate reconstruction loss (log_prob of inputs under px_z)
        reconstruction = px_z.log_prob(inputs)
        # TODO: Calculate KL divergence between qz_x and prior (pz)
        divergence = qz_x.kl_divergence(self.pz)
        # TODO: Calculate ELBO (reconstruction - KL)
        ELBO = reconstruction - divergence

        # TODO: Set self.loss to the negative mean of ELBO
        self.loss = -tf.reduce_mean(ELBO)
        # TODO: Return the mean ELBO
        return tf.reduce_mean(ELBO)

    @tf.function
    def train(self, x):
        # TODO: Open a tf.GradientTape()
        with tf.GradientTape() as tape:
            # TODO: Call the model on x to get the ELBO
            elbo = self.call(x)
        # TODO: Calculate gradients of self.loss with respect to self.trainable_variables
        grads = tape.gradient(self.loss, self.trainable_variables)
        # TODO: Apply gradients using self.optimizer
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # TODO: Return the ELBO
        return elbo

    def serving_reconstruct(self, inputs, L):
        """
        Performs Monte Carlo sampling to calculate true ERE probability density.
        """
        all_px_z = tf.TensorArray(dtype=tf.float32, size=L)
        for i in tf.range(L):
            z = self.encoder(inputs).sample()
            log_px_z = self.decoder(z).log_prob(inputs)
            all_px_z = all_px_z.write(i, log_px_z)

        stacked = tf.math.exp(all_px_z.stack())
        ere = tf.reduce_mean(stacked, axis=0)

        return {"ere": ere}

    def generating_latent(self, inputs):
        """
        Draw latent variables z given inputs (Data Distribution Shifts exercise)
        """
        z_sample = self.encoder(inputs).sample()

        return {"z_sample": z_sample}
