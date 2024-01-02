import tensorflow as tf
from tensorflow.keras.activations import gelu
from keras import layers
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
)


def transformer_block(
    inputs: layers.Input,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    att_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    explainable: bool = False,
    post_norm: bool = True,        
):
    # Define the layers
    att = MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim, dropout=att_dropout
    )
    skip1 = Add()
    layernorm1 = LayerNormalization(epsilon=1e-6)
    ffn = tf.keras.Sequential(
        [
            Dense(ff_dim, activation=gelu),
            Dropout(ff_dropout),
            Dense(embed_dim),
        ]
    )
    layernorm2 = LayerNormalization(epsilon=1e-6)
    skip2 = Add()    
    

    # Post-norm variant
    if post_norm:
        inputs = layernorm1(inputs)
        if explainable:
            attention_output, att_weights = att(
                inputs, inputs, return_attention_scores=True
            )
        else:
            attention_output = att(inputs, inputs)
        attention_output = skip1([inputs, attention_output])   
        feedforward_output = ffn(attention_output) 
        transformer_output = skip2([feedforward_output, attention_output])
        transformer_output = layernorm2(transformer_output)
    # Pre-norm variant
    else:
        norm_input = layernorm1(inputs)
        if explainable:
            attention_output, att_weights = att(
                norm_input, norm_input, return_attention_scores=True
            )
        else:
            attention_output = att(norm_input, norm_input)

        attention_output = skip1([inputs, attention_output])
        norm_attention_output = layernorm2(attention_output)
        feedforward_output = ffn(norm_attention_output)
        transformer_output = skip2([feedforward_output, attention_output])
    
    if explainable:
        return transformer_output, att_weights
    else:
        return transformer_output
