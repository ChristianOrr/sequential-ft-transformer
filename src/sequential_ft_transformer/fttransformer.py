import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    LayerNormalization
)

from sequential_ft_transformer.transformer import transformer_block
from sequential_ft_transformer.embeddings import (
    cat_embedding,
    num_embedding,
)


def ft_transformer_encoder(
    numeric_inputs: layers.Input,
    cat_inputs: layers.Input,
    categorical_features: list,
    numerical_features: list,
    feature_unique_counts: list,
    seq_length: int = 1,
    embedding_dim: int = 32,
    depth: int = 4,
    heads: int = 8,
    attn_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    numerical_embedding_type: str = 'linear',
    bins_dict: dict = None,
    n_bins: int = None,
    explainable: bool = False,       
):
    
    w_init = tf.random_normal_initializer()
    if numeric_inputs is not None:
        batch_size = tf.shape(numeric_inputs)[0]
    elif cat_inputs is not None:
        batch_size = tf.shape(cat_inputs)[0]
    else:
        raise ValueError("Numeric and Categorical inputs are None. Need at least one.")

    shape = tf.stack([batch_size, seq_length, 1, embedding_dim], axis=0)
    cls_weights = w_init(shape)
    transformer_inputs = [cls_weights]

    if categorical_features is not None and cat_inputs is not None:
        cat_embs = cat_embedding(
            inputs=cat_inputs,
            feature_names=categorical_features,
            feature_unique_counts=feature_unique_counts,
            emb_dim =embedding_dim
        )
        transformer_inputs += [cat_embs]
    
    if numerical_features is not None and numeric_inputs is not None:
        num_embs = num_embedding(
            inputs=numeric_inputs,
            batch_size=batch_size,
            feature_names=numerical_features, 
            seq_length=seq_length,
            emb_dim=embedding_dim, 
            emb_type=numerical_embedding_type, 
            bins_dict=bins_dict, 
            n_bins=n_bins,
        )

        transformer_inputs += [num_embs]

    transformer_inputs = tf.concat(transformer_inputs, axis=2)
    importances = []

    # Pass through Transformer blocks
    for _ in range(depth):
        if explainable:
            transformer_inputs, att_weights = transformer_block(
                transformer_inputs,
                embedding_dim,
                heads,
                embedding_dim,
                att_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                explainable=explainable,
                post_norm=False,  # FT-Transformer uses pre-norm
            )
            att = att_weights[:, :, :, 0, :, :]
            att = tf.reduce_sum(att, axis=(1, 2, 3))
            importances.append(att)
        else:
            transformer_inputs = transformer_block(
                transformer_inputs,
                embedding_dim,
                heads,
                embedding_dim,
                att_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                explainable=explainable,
                post_norm=False,  # FT-Transformer uses pre-norm
            )

    if explainable:
        importances = tf.reduce_sum(tf.stack(importances), axis=0) / (
            depth * heads
        )
        return transformer_inputs, importances
    else:
        return transformer_inputs, None
        



def ft_transformer(
    out_dim: int,
    out_activation: str,
    feature_unique_counts: list,
    categorical_features: list = None,
    numerical_features: list = None,
    seq_length: int = 1,
    embedding_dim: int = 32,
    depth: int = 4,
    heads: int = 8,
    attn_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    numerical_embedding_type: str = 'linear',
    bins_dict: dict = None,
    n_bins: int = None,
    explainable: bool = False,      
):
    
    ln = LayerNormalization()
    flatten = Flatten()
    dense_dim_size = embedding_dim * seq_length
    dense1 = Dense(dense_dim_size//2, activation='relu')
    dense2 = Dense(dense_dim_size//4, activation='relu')
    output_layer = Dense(out_dim, activation=out_activation, name="output")

    inputs_dict = dict()
    empty_cat: bool = categorical_features is None
    empty_numeric: bool = numerical_features is None

    if empty_cat and empty_numeric:
        raise ValueError("Both categorical and numerical features are missing. At least one is needed")
    
    if not empty_numeric:
        numeric_input_shape = (seq_length, len(numerical_features), )
        numeric_inputs = layers.Input(shape=numeric_input_shape, name="numeric_inputs")
        inputs_dict.update({"numeric_inputs": numeric_inputs})
    else:
        numeric_inputs = None

    if not empty_cat:
        cat_input_shape = (seq_length, len(categorical_features), )
        cat_inputs = layers.Input(shape=cat_input_shape, name="cat_inputs")
        inputs_dict.update({"cat_inputs": cat_inputs})
    else:
        cat_inputs = None

    x, expl = ft_transformer_encoder(
        numeric_inputs=numeric_inputs,
        cat_inputs=cat_inputs,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        feature_unique_counts=feature_unique_counts,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        numerical_embedding_type=numerical_embedding_type,
        bins_dict=bins_dict,
        n_bins=n_bins,
        explainable=explainable,
    )

    layer_norm_cls = ln(x[:, :, 0, :])
    layer_norm_cls = flatten(layer_norm_cls)
    layer_norm_cls = dense1(layer_norm_cls)
    layer_norm_cls = dense2(layer_norm_cls)
    output = output_layer(layer_norm_cls)

    outputs_dict = {"output": output}
    if explainable:
        outputs_dict.update({"importances": expl})

    model = tf.keras.Model(inputs=inputs_dict,
                           outputs=outputs_dict,
                           name="FT-Transformer")

    return model

