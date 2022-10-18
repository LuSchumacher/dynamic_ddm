import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.models import Sequential
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers


class DynamicGaussianNetwork(tf.keras.Model):
    """TODO: Logic"""

    def __init__(self, meta):
        super(DynamicGaussianNetwork, self).__init__()

        self.embedding_net = Sequential([
            LSTM(meta['embedding_lstm_units'], return_sequences=True),
            Dense(**meta['dense_pre_args']),
        ])

        self.micro_part = Sequential([
            Dense(**meta['dense_micro_args']),
            Dense(tfpl.MultivariateNormalTriL.params_size(meta['n_micro_params'])),
            tfpl.MultivariateNormalTriL(meta['n_micro_params'])
        ])

        self.macro_part = Sequential([
            LSTM(meta['macro_lstm_units']),
            Dense(tfpl.IndependentNormal.params_size(meta['n_macro_params'])),
            tfpl.IndependentNormal(meta['n_macro_params'])
        ])

    def call(self, data, macro_params):
        """ Performs a forward pass through the network and obtains the posterior(s) over all time points.

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)
        macro_params : tf.Tensor of np.ndarray of shape (batch_size, n_macro_params)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)

        # Predict dynamic microscopic params
        micro_params_hat = self.micro_part(
            tf.concat([rep, tf.stack([macro_params] * T, axis=1)], axis=-1)
        )

        return macro_params_hat, micro_params_hat


    def sample(self, data):
        """ Performs a forward pass through the network and obtains the posterior(s) over all time points.

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)
        macro_params_rv = macro_params_hat.sample()

        # Predict dynamic microscopic params
        micro_params_hat = self.micro_part(
            tf.concat([rep, tf.stack([macro_params_rv] * T, axis=1)], axis=-1)
        )
        micro_params_rv = micro_params_hat.sample()

        return macro_params_rv, micro_params_rv


    def sample_n(self, data, n_samples=50):
        """ TODO

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)

        # Prepare placeholders
        macro_samples = [None] * n_samples
        micro_samples = [None] * n_samples
        for n in range(n_samples):

            macro_params_rv = macro_params_hat.sample()
            micro_params_hat = self.micro_part(
                tf.concat([rep, tf.stack([macro_params_rv] * T, axis=1)], axis=-1)
            )
            micro_params_rv = micro_params_hat.sample()
            macro_samples[n] = macro_params_rv
            micro_samples[n] = micro_params_rv
        return tf.stack(macro_samples), tf.stack(micro_samples)


    def sample_n_fixed_eta(self, data, n_samples=50):
        """ TODO

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)

        # Prepare placeholders
        macro_samples = [None] * n_samples
        micro_samples = [None] * n_samples
        for n in range(n_samples):

            macro_params_rv = macro_params_hat.sample()
            macro_params_rv = macro_params_rv.numpy()
            macro_params_rv[:, :] = 0
            micro_params_hat = self.micro_part(
                tf.concat([rep, tf.stack([macro_params_rv] * T, axis=1)], axis=-1)
            )
            micro_params_rv = micro_params_hat.sample()
            macro_samples[n] = macro_params_rv
            micro_samples[n] = micro_params_rv
        return tf.stack(macro_samples), tf.stack(micro_samples)


class DynamicGaussianNetwork_legacy(tf.keras.Model):
    """TODO: Logic"""

    def __init__(self, meta):
        super(DynamicGaussianNetwork_legacy, self).__init__()

        self.embedding_net = Sequential([
            LSTM(meta['embedding_lstm_units'], return_sequences=True),
            GRU(meta['embedding_gru_units'], return_sequences=True),
            Dense(**meta['dense_pre_args']),
        ])

        self.micro_part = Sequential([
            Dense(**meta['dense_micro_args']),
            Dense(tfpl.MultivariateNormalTriL.params_size(meta['n_micro_params'])),
            tfpl.MultivariateNormalTriL(meta['n_micro_params'])
        ])

        self.macro_part = Sequential([
            LSTM(meta['macro_lstm_units']),
            Dense(tfpl.IndependentNormal.params_size(meta['n_macro_params'])),
            tfpl.IndependentNormal(meta['n_macro_params'])
        ])

    def call(self, data, macro_params):
        """ Performs a forward pass through the network and obtains the posterior(s) over all time points.

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)
        macro_params : tf.Tensor of np.ndarray of shape (batch_size, n_macro_params)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)

        # Predict dynamic microscopic params
        micro_params_hat = self.micro_part(
            tf.concat([rep, tf.stack([macro_params] * T, axis=1)], axis=-1)
        )

        return macro_params_hat, micro_params_hat


    def sample(self, data):
        """ Performs a forward pass through the network and obtains the posterior(s) over all time points.

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)
        macro_params_rv = macro_params_hat.sample()

        # Predict dynamic microscopic params
        micro_params_hat = self.micro_part(
            tf.concat([rep, tf.stack([macro_params_rv] * T, axis=1)], axis=-1)
        )
        micro_params_rv = micro_params_hat.sample()

        return macro_params_rv, micro_params_rv


    def sample_n(self, data, n_samples=50):
        """ TODO

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)

        # Prepare placeholders
        macro_samples = [None] * n_samples
        micro_samples = [None] * n_samples
        for n in range(n_samples):

            macro_params_rv = macro_params_hat.sample()
            micro_params_hat = self.micro_part(
                tf.concat([rep, tf.stack([macro_params_rv] * T, axis=1)], axis=-1)
            )
            micro_params_rv = micro_params_hat.sample()
            macro_samples[n] = macro_params_rv
            micro_samples[n] = micro_params_rv
        return tf.stack(macro_samples), tf.stack(micro_samples)


    def sample_n_fixed_eta(self, data, n_samples=50):
        """ TODO

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)

        # Prepare placeholders
        macro_samples = [None] * n_samples
        micro_samples = [None] * n_samples
        for n in range(n_samples):

            macro_params_rv = macro_params_hat.sample()
            macro_params_rv = macro_params_rv.numpy()
            macro_params_rv[:, :] = 0
            micro_params_hat = self.micro_part(
                tf.concat([rep, tf.stack([macro_params_rv] * T, axis=1)], axis=-1)
            )
            micro_params_rv = micro_params_hat.sample()
            macro_samples[n] = macro_params_rv
            micro_samples[n] = micro_params_rv
        return tf.stack(macro_samples), tf.stack(micro_samples)