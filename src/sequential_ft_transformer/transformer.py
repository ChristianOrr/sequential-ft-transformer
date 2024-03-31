import keras
from keras.layers import (
    Add,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
)


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, attn_dropout=0.1, ff_dropout=0.1, explainable=False, post_norm=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.explainable = explainable
        self.post_norm = post_norm

        # Layers within the block
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=attn_dropout)
        self.skip1 = Add()
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="gelu"), Dropout(ff_dropout), Dense(embed_dim)]
        )
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.skip2 = Add()

    def call(self, inputs):
        if self.post_norm:
            normalized_inputs = self.layernorm1(inputs)
            attention_output = self.att(normalized_inputs, normalized_inputs)
            attention_output = self.skip1([inputs, attention_output])
            ff_output = self.ffn(attention_output)
            output = self.skip2([ff_output, attention_output])
            output = self.layernorm2(output)
        else:
            normalized_inputs = self.layernorm1(inputs)
            attention_output = self.att(normalized_inputs, normalized_inputs)
            attention_output = self.skip1([inputs, attention_output])
            normalized_att_output = self.layernorm2(attention_output)
            ff_output = self.ffn(normalized_att_output)
            output = self.skip2([ff_output, attention_output])

        if self.explainable:
            return output, self.att.attention_weights  # Access attention weights from MultiHeadAttention
        else:
            return output