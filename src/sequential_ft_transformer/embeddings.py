import tensorflow as tf
import keras
from keras import layers
import numpy as np


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


def ple(
    inputs: layers.Input,
    feature_names: list,
    bins_dict: dict, 
    seq_length: int,   
    emb_dim: int,
    dtype: str = "float32"    
):
    """
    Creates a layer that applies PLE to multiple numerical features.

    Args:
        inputs: A TensorFlow tensor of shape (batch_size, seq_length, num_features) representing numerical features.
        batch_size: The batch size.
        feature_names: A list of feature names.
        bins_dict: A dictionary mapping feature names to their bin boundaries.
        seq_length: The length of the input sequence.
        emb_dim: The embedding dimension.

    Returns:
        A TensorFlow tensor of shape (batch_size, seq_length, num_features, emb_dim) containing the embedded features.
    """

    emb_columns = []
    for i, f in enumerate(feature_names):
        
        # We need to use Tensorflow here since there is no unique operation in keras 3
        bins = tf.unique(bins_dict[f]).y
        bins = keras.ops.cast(bins, dtype)
        
        ple_feature_layer = PLEFeatureLayer(bins=bins, seq_length=seq_length, dtype=dtype)
        emb_l = ple_feature_layer(inputs[:, :, i])
        lin_l = keras.layers.Dense(emb_dim, activation='relu')
        
        embedded_col = lin_l(emb_l)
        emb_columns.append(embedded_col)

    embs = keras.ops.concatenate(emb_columns, axis=2)

    return embs 


class PeriodicEmbedding(keras.layers.Layer):
    def __init__(self, emb_dim, seq_length, num_features, n_bins, sigma):
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


class LinearEmbedding(keras.layers.Layer):
    def __init__(self, emb_dim, seq_length, num_features):
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


def numeric_embedding(
    inputs: layers.Input,
    feature_names: list,
    seq_length: int,
    emb_dim: int,
    emb_type: str = 'linear',
    bins_dict: dict = None,
    n_bins: int = None,
    sigma: float = 1,    
):
    """
    Creates numerical embeddings using the specified embedding type.

    Args:
        inputs: A TensorFlow tensor of shape (batch_size, seq_length, num_features) representing numerical features.
        feature_names: A list of feature names.
        seq_length: The length of the input sequence.
        emb_dim: The embedding dimension.
        batch_size: The batch size (optional, required for PLE).
        emb_type: The type of numerical embedding to use ('linear', 'ple', or 'periodic').
        bins_dict: A dictionary mapping feature names to their bin boundaries (required for PLE).
        n_bins: The number of bins for periodic encoding (required for periodic encoding).
        sigma: The standard deviation for weight initialization (used for periodic encoding).

    Returns:
        A TensorFlow tensor of shape (batch_size, seq_length, num_features, emb_dim) containing the embedded features.
    """
    num_features = len(feature_names)

    if emb_type == 'ple':
        if bins_dict is None:
            raise ValueError(f"bins_dict is required for ple numerical embedding, received: {bins_dict}")
        embs = ple(
            inputs=inputs,
            feature_names=feature_names,
            bins_dict=bins_dict, 
            seq_length=seq_length,   
            emb_dim=emb_dim    
        )

    elif emb_type == 'periodic':
        if n_bins is None:
            raise ValueError(f"n_bins is required for periodic numerical embedding, received: {n_bins}")
        embs = PeriodicEmbedding(emb_dim, seq_length, num_features, n_bins, sigma)(inputs)
        
    elif emb_type == 'linear':
        embs = LinearEmbedding(emb_dim, seq_length, num_features)(inputs)
         
    else:
        raise ValueError(f"emb_type: {emb_type} is not supported")  
      
    return embs
    

def cat_embedding(
    inputs: layers.Input,
    feature_names: list,
    feature_unique_counts: dict,
    emb_dim: int,
):
    """
    Creates categorical embeddings for a set of categorical features.

    Args:
        inputs: A TensorFlow tensor of shape (batch_size, seq_length, num_features) representing categorical features.
        feature_names: A list of feature names.
        feature_unique_counts: A dictionary mapping feature names to their number of unique values.
        emb_dim: The embedding dimension.

    Returns:
        A TensorFlow tensor of shape (batch_size, seq_length, num_features, emb_dim) containing the embedded features.
    """
    emb_layers = {}
    for cat_name, unique_count in feature_unique_counts.items():
        emb = keras.layers.Embedding(input_dim=unique_count, output_dim=emb_dim)
        emb_layers[cat_name] = emb

    emb_columns = []
    for i, f in enumerate(feature_names):
        embedded_col = emb_layers[f](inputs[:, :, i])
        emb_columns.append(embedded_col)

    embs = keras.ops.stack(emb_columns, axis=2)
    return embs
