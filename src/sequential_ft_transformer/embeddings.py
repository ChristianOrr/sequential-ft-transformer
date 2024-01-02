import tensorflow as tf
from keras import layers
import numpy as np
import math as m


def ple(
    inputs: layers.Input,
    batch_size: int,
    bins: tf.TensorArray, 
    seq_length: int,
):
    """
    Creates piecewise linear encoding (PLE) for numerical features.

    Args:
        inputs: A TensorFlow tensor of shape (batch_size, seq_length, 1) representing a numerical feature.
        batch_size: The batch size.
        bins: A TensorFlow tensor representing the bin boundaries for the feature.
        seq_length: The length of the input sequence.

    Returns:
        A TensorFlow tensor of shape (batch_size, seq_length, 1, n_bins) containing the PLE encoding.
    """
    n_bins = len(bins)
    lookup_keys = [i for i in range(n_bins)]
    init = tf.lookup.KeyValueTensorInitializer(lookup_keys, bins)
    lookup_table = tf.lookup.StaticHashTable(init, default_value=-1)

    ple_encoding_one = tf.ones((batch_size, seq_length, n_bins))
    ple_encoding_zero = tf.zeros((batch_size, seq_length, n_bins))

    left_masks = []
    right_masks = []
    other_case = []

    for i in range(1, n_bins + 1):
        i = tf.constant(i)
        left_mask = (inputs < lookup_table.lookup(i - 1)) & (i > 1)
        right_mask = (inputs >= lookup_table.lookup(i)) & (i < n_bins)
        v = (inputs - lookup_table.lookup(i - 1)) / (
            lookup_table.lookup(i) - lookup_table.lookup(i - 1)
        )
        left_masks.append(left_mask)
        right_masks.append(right_mask)
        other_case.append(v)

    left_masks = tf.stack(left_masks, axis=2)
    right_masks = tf.stack(right_masks, axis=2)
    other_case = tf.stack(other_case, axis=2)     

    other_mask = right_masks == left_masks  # both are false
    other_case = tf.cast(other_case, tf.float32)
    enc = tf.where(left_masks, ple_encoding_zero, ple_encoding_one)
    enc = tf.where(other_mask, other_case, enc)
    enc = tf.reshape(enc, (-1, seq_length, 1, n_bins))

    return enc


def ple_layer(
    inputs: layers.Input,
    batch_size: int,
    feature_names: list,
    bins_dict: dict, 
    seq_length: int,   
    emb_dim: int     
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
        
        bins = tf.cast(bins_dict[f], tf.float32)
        bins = tf.unique(bins).y

        emb_l = ple(inputs[:, :, i], batch_size, bins, seq_length)
        lin_l = tf.keras.layers.Dense(emb_dim, activation='relu')
        
        embedded_col = lin_l(emb_l)
        emb_columns.append(embedded_col)

    embs = tf.concat(emb_columns, axis=2)

    return embs


def periodic(
    inputs: layers.Input,
    emb_dim: int,
    seq_length: int, 
    num_features: int,
    n_bins: int, 
    sigma: float,
):
    """
    Creates periodic encoding for numerical features.

    Args:
        inputs: A TensorFlow tensor of shape (batch_size, seq_length, num_features) representing numerical features.
        emb_dim: The embedding dimension.
        seq_length: The length of the input sequence.
        num_features: The number of features.
        n_bins: The number of bins for the periodic encoding.
        sigma: The standard deviation for weight initialization.

    Returns:
        A TensorFlow tensor of shape (batch_size, seq_length, num_features, emb_dim) containing the embedded features.
    """
    w_init = tf.random_normal_initializer(stddev=sigma)
    p_shape = (seq_length, num_features, n_bins)
    p = tf.Variable(
        initial_value=w_init(
            shape=p_shape,
            dtype='float32'),
        trainable=True)

    l_shape = (seq_length, num_features, n_bins*2, emb_dim)
    l = tf.Variable(
        initial_value=w_init(
            shape=l_shape, 
            dtype='float32' 
            ), trainable=True)

    v = 2 * m.pi * p[None] * inputs[..., None]
    emb = tf.concat([tf.math.sin(v), tf.math.cos(v)], axis=-1)
    emb = tf.einsum('sfne, bsfn -> bsfe', l, emb)
    emb = tf.nn.relu(emb)
    return emb


def linear(
    inputs: layers.Input,
    emb_dim: int,
    seq_length: int, 
    num_features: int       
):
    """
    Creates a linear embedding for numerical features.

    Args:
        inputs: A TensorFlow tensor of shape (batch_size, seq_length, num_features) representing numerical features.
        emb_dim: The embedding dimension.
        seq_length: The length of the input sequence.
        num_features: The number of features.

    Returns:
        A TensorFlow tensor of shape (batch_size, seq_length, num_features, emb_dim) containing the embedded features.
    """
    w_init = tf.random_normal_initializer()
    linear_w = tf.Variable(
        initial_value=w_init(
            shape=(seq_length, num_features, 1, emb_dim), dtype='float32' # features, n_bins, emb_dim
        ), trainable=True)
    linear_b = tf.Variable(
        w_init(
            shape=(seq_length, num_features, 1), dtype='float32' # features, n_bins, emb_dim
        ), trainable=True)
    
    embs = tf.einsum('sfne, bsf -> bsfe', linear_w, inputs)
    embs = tf.nn.relu(embs + linear_b) 

    return embs    
  

def numeric_embedding(
    inputs: layers.Input,
    feature_names: list,
    seq_length: int,
    emb_dim: int,
    batch_size: int = None,
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
        if batch_size is None:
            raise ValueError(f"batch_size is required for ple numerical embedding, received: {batch_size}")
        embs = ple_layer(
            inputs=inputs,
            batch_size=batch_size,
            feature_names=feature_names,
            bins_dict=bins_dict, 
            seq_length=seq_length,   
            emb_dim=emb_dim    
        )

    elif emb_type == 'periodic':
        if n_bins is None:
            raise ValueError(f"n_bins is required for periodic numerical embedding, received: {n_bins}")
        embs = periodic(
            inputs=inputs,
            n_bins=n_bins,
            seq_length=seq_length,
            num_features=num_features,
            emb_dim=emb_dim,
            sigma=sigma)
        
    elif emb_type == 'linear':
        embs = linear(
            inputs=inputs,
            seq_length=seq_length,
            num_features=num_features,
            emb_dim=emb_dim) 
         
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
        emb = tf.keras.layers.Embedding(input_dim=unique_count, output_dim=emb_dim)
        emb_layers[cat_name] = emb

    emb_columns = []
    for i, f in enumerate(feature_names):
        embedded_col = emb_layers[f](inputs[:, :, i])
        emb_columns.append(embedded_col)

    embs = tf.stack(emb_columns, axis=2)
    return embs
