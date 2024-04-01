import tensorflow as tf
import keras
from keras import layers
import numpy as np


@keras.saving.register_keras_serializable(package="TransformerLayers")
class PLEFeatureLayer(keras.layers.Layer):
    """
    Creates piecewise linear encoding (PLE) for a numerical feature.

    Args:
        inputs: A TensorFlow tensor of shape (batch_size, seq_length, 1) representing a numerical feature.
        batch_size: The batch size.
        bins: A TensorFlow tensor representing the bin boundaries for the feature.
        seq_length: The length of the input sequence.

    Returns:
        A TensorFlow tensor of shape (batch_size, seq_length, 1, n_bins) containing the PLE encoding.
    """
    def __init__(self, bins, seq_length, dtype="float32", **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        self.n_bins = len(bins)
        self.seq_length = seq_length
        self.type = dtype

    def call(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]

        ple_encoding_one = keras.ops.ones((batch_size, self.seq_length, self.n_bins))
        ple_encoding_zero = keras.ops.zeros((batch_size, self.seq_length, self.n_bins))

        left_masks = []
        right_masks = []
        other_case = []

        for i in range(1, self.n_bins + 1):
            left_mask = (inputs < self.bins[i - 1]) & (i > 1)
            right_mask = (inputs >= self.bins[i]) & (i < self.n_bins)
            v = (inputs - self.bins[i - 1]) / (
                self.bins[i] - self.bins[i - 1] + 1e-8 # Add a small value to prevent division by zero
            )
            left_masks.append(left_mask)
            right_masks.append(right_mask)
            other_case.append(v)

        left_masks = keras.ops.stack(left_masks, axis=2)
        right_masks = keras.ops.stack(right_masks, axis=2)
        other_case = keras.ops.stack(other_case, axis=2)     

        other_mask = right_masks == left_masks  # both are false
        other_case = keras.ops.cast(other_case, self.type)
        enc = keras.ops.where(left_masks, ple_encoding_zero, ple_encoding_one)
        enc = keras.ops.where(other_mask, other_case, enc)
        enc = keras.ops.reshape(enc, (-1, self.seq_length, 1, self.n_bins))

        return enc

    def get_config(self):
        config = {
            "bins": self.bins,
            "seq_length": self.seq_length,
            "dtype": self.dtype,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="TransformerLayers")
class PLELayer(keras.layers.Layer):
    def __init__(self, feature_names, bins_dict, seq_length, emb_dim, dtype="float32", **kwargs):
        super().__init__(**kwargs)
        self.feature_names = feature_names
        self.bins_dict = bins_dict
        self.seq_length = seq_length
        self.emb_dim = emb_dim
        self.type = dtype

        self.ple_layers = {}
        self.dense_layers = {}

    def build(self, input_shape):
        for feature_name in self.feature_names:
            # We need to use Tensorflow here since there is no unique operation in keras 3
            bins = tf.unique(self.bins_dict[feature_name]).y
            bins = keras.ops.cast(bins, self.dtype)
            self.ple_layers[feature_name] = PLEFeatureLayer(bins=bins, seq_length=self.seq_length, dtype=self.dtype)
            self.dense_layers[feature_name] = keras.layers.Dense(self.emb_dim, activation="relu") 

    def call(self, inputs):
        embedded_features = []
        for i, feature_name in enumerate(self.feature_names):
            # Use pre-built PLEFeatureLayer for each feature
            ple_layer = self.ple_layers[feature_name]
            ple_encoding = ple_layer(inputs[:, :, i])  

            # Add dense layer for final embedding
            dense_layer = self.dense_layers[feature_name]
            embedded_feature = dense_layer(ple_encoding)
            embedded_features.append(embedded_feature)

        # Concatenate embedded features along the feature dimension
        embedded_features = keras.ops.concatenate(embedded_features, axis=2)
        return embedded_features

    def get_config(self):
        config = {
            "feature_names": self.feature_names,
            "bins_dict": self.bins_dict,  # Include for deserialization
            "seq_length": self.seq_length,
            "emb_dim": self.emb_dim,
            "dtype": self.dtype,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="TransformerLayers")
class PeriodicEmbedding(keras.layers.Layer):
    def __init__(self, emb_dim, seq_length, num_features, n_bins, sigma, **kwargs):
        super().__init__()
        self.emb_dim = emb_dim
        self.seq_length = seq_length
        self.num_features = num_features
        self.n_bins = n_bins
        self.sigma = sigma

        self.p = None
        self.l = None

    def build(self, input_shape):
        w_init = keras.initializers.RandomNormal(stddev=self.sigma)
        self.p = self.add_weight(
            shape=(self.seq_length, self.num_features, self.n_bins),
            initializer=w_init,
            trainable=True,
            name='p'
        )
        self.l = self.add_weight(
            shape=(self.seq_length, self.num_features, self.n_bins * 2, self.emb_dim),
            initializer=w_init,
            trainable=True,
            name='l'
        )

    def call(self, inputs):
        v = 2 * np.pi * self.p[None] * inputs[..., None]
        embs = keras.ops.concatenate([keras.ops.sin(v), keras.ops.cos(v)], axis=-1)
        embs = keras.ops.einsum('sfne, bsfn -> bsfe', self.l, embs)
        embs = keras.ops.relu(embs)
        return embs
    
    def get_config(self):
        config = {
            "emb_dim": self.emb_dim,
            "seq_length": self.seq_length,
            "num_features": self.num_features,
            "n_bins": self.n_bins,
            "sigma": self.sigma,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="TransformerLayers")
class LinearEmbedding(keras.layers.Layer):
    def __init__(self, emb_dim, seq_length, num_features, **kwargs):
        super().__init__()
        self.emb_dim = emb_dim
        self.seq_length = seq_length
        self.num_features = num_features

    def build(self, input_shape):
        self.linear_w = self.add_weight(
            shape=(self.seq_length, self.num_features, 1, self.emb_dim),
            initializer='random_normal',
            trainable=True,
            name='linear_w'
        )
        self.linear_b = self.add_weight(
            shape=(self.seq_length, self.num_features, 1),
            initializer='random_normal',
            trainable=True,
            name='linear_b'
        )

    def call(self, inputs):
        embs = keras.ops.einsum('sfne, bsf -> bsfe', self.linear_w, inputs)
        embs = keras.ops.relu(embs + self.linear_b)
        return embs

    def get_config(self):
        config = {
            "emb_dim": self.emb_dim,
            "seq_length": self.seq_length,
            "num_features": self.num_features,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

@keras.saving.register_keras_serializable(package="TransformerLayers")
class NumericEmbeddingLayer(keras.layers.Layer):
    def __init__(self, feature_names, seq_length, emb_dim, emb_type="linear", bins_dict=None, n_bins=None, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.feature_names = feature_names
        self.seq_length = seq_length
        self.emb_dim = emb_dim
        self.emb_type = emb_type
        self.bins_dict = bins_dict
        self.n_bins = n_bins
        self.sigma = sigma
        self.supported_layers = ["ple", "periodic", "linear"]

        # Validate arguments based on emb_type
        self._validate_arguments()
        # Internal layers (optional depending on emb_type)
        self.periodic_layer = None
        self.linear_layer = None

    def _validate_arguments(self):
        if self.emb_type not in self.supported_layers:
            raise ValueError(f"emb_type: {self.emb_type} is not supported. Use one of the following layers: {self.supported_layers}")
        if self.emb_type == "ple" and self.bins_dict is None:
            raise ValueError("bins_dict is required for PLE embedding.")
        elif self.emb_type == "periodic" and self.n_bins is None:
            raise ValueError("n_bins is required for periodic embedding.")

    def build(self, input_shape):
        if self.emb_type == "periodic":
            self.periodic_layer = PeriodicEmbedding(
                self.emb_dim, self.seq_length, len(self.feature_names), self.n_bins, self.sigma
            )
        elif self.emb_type == "linear":
            self.linear_layer = LinearEmbedding(self.emb_dim, self.seq_length, len(self.feature_names))

    def call(self, inputs):
        if self.emb_type == "ple":
            # The PLE Layer needs to be in the call, 
            # since the batch size is used from inputs
            ple_layer = PLELayer(
                feature_names=self.feature_names,
                bins_dict=self.bins_dict,
                seq_length=self.seq_length,
                emb_dim=self.emb_dim,
                dtype="float32",
            )
            embeddings = ple_layer(inputs)
        elif self.emb_type == "periodic":
            embeddings = self.periodic_layer(inputs)
        else:
            embeddings = self.linear_layer(inputs)

        return embeddings

    def get_config(self):
        config = {
            "feature_names": self.feature_names,
            "seq_length": self.seq_length,
            "emb_dim": self.emb_dim,
            "emb_type": self.emb_type,
            "bins_dict": self.bins_dict,
            "n_bins": self.n_bins,
            "sigma": self.sigma,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

@keras.saving.register_keras_serializable(package="TransformerLayers")
class CatEmbeddingLayer(keras.layers.Layer):
    def __init__(self, feature_unique_counts, emb_dim, **kwargs):
        super().__init__(**kwargs)
        self.feature_unique_counts = feature_unique_counts
        self.emb_dim = emb_dim

        self.emb_layers = {}

    def build(self, input_shape):
        for feature_name, unique_count in self.feature_unique_counts.items():
            emb = keras.layers.Embedding(input_dim=unique_count, output_dim=self.emb_dim)
            self.emb_layers[feature_name] = emb

    def call(self, inputs):
        # Embed each categorical feature using corresponding layer
        emb_columns = []
        for i, f in enumerate(self.feature_unique_counts.keys()):
            embedded_feature = self.emb_layers[f](inputs[:, :, i])
            emb_columns.append(embedded_feature)

        embs = keras.ops.stack(emb_columns, axis=2)
        return embs
    
    def get_config(self):
        config = {
            "feature_unique_counts": self.feature_unique_counts,
            "emb_dim": self.emb_dim,
            "emb_layers": self.emb_layers,  # Include serialized embedding layers
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)