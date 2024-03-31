import keras
from sequential_ft_transformer.transformer import TransformerBlock
from sequential_ft_transformer.embeddings import (
    CatEmbeddingLayer,
    NumericEmbeddingLayer,
)
from typing import List  


class CLSWeightsLayer(keras.layers.Layer):
  def __init__(self, seq_length, embedding_dim, **kwargs):
    super().__init__(**kwargs)
    self.seq_length = seq_length
    self.embedding_dim = embedding_dim

  def build(self, input_shape):
    # Initial shape with single dimension for batch size
    self.cls_weights = self.add_weight(
        shape=[1, self.seq_length, 1, self.embedding_dim],
        initializer="random_normal",
        trainable=True
    )

  def call(self, inputs):
    batch_size = keras.ops.shape(inputs)[0]
    # Repeat the cls_weights to match the batch_size
    cls_weights_tiled = keras.ops.tile(self.cls_weights, batch_size)
    cls_weights = keras.ops.reshape(cls_weights_tiled, (batch_size, self.seq_length, 1, self.embedding_dim))
    return cls_weights


class FTTransformerEncoder(keras.layers.Layer):
    def __init__(self,
                 categorical_features,
                 numerical_features,
                 feature_unique_counts,
                 seq_length=1,
                 embedding_dim=16,
                 depth=4,
                 heads=8,
                 attn_dropout=0.2,
                 ff_dropout=0.2,
                 numerical_embedding_type="linear",
                 bins_dict=None,
                 n_bins=None,
                 explainable=False,
                 post_norm=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.feature_unique_counts = feature_unique_counts
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.numerical_embedding_type = numerical_embedding_type
        self.bins_dict = bins_dict
        self.n_bins = n_bins
        self.explainable = explainable

        # Internal layers
        self.cls_weights_layer = CLSWeightsLayer(seq_length, embedding_dim)
        if categorical_features:
            self.cat_layer = CatEmbeddingLayer(
                feature_unique_counts=feature_unique_counts, emb_dim=embedding_dim
            )
        if numerical_features:
            self.numeric_layer = NumericEmbeddingLayer(
                feature_names=numerical_features,
                seq_length=seq_length,
                emb_dim=embedding_dim,
                emb_type=numerical_embedding_type,
                bins_dict=bins_dict,
                n_bins=n_bins,
            )
        self.transformer_blocks = [
            TransformerBlock(
                embedding_dim,
                heads,
                embedding_dim,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                explainable=explainable,
                post_norm=post_norm,
            )
            for _ in range(depth)
        ]

    def call(self, inputs):
        numeric_inputs, cat_inputs = inputs["numeric_inputs"], inputs["cat_inputs"]
        # Process CLS weights
        cls_weights = self.cls_weights_layer(numeric_inputs if numeric_inputs is not None else cat_inputs)

        # Embedding layers
        embeddings = [cls_weights]
        if self.categorical_features:
            cat_embs = self.cat_layer(cat_inputs)
            embeddings.append(cat_embs)
        if self.numerical_features:
            num_embs = self.numeric_layer(numeric_inputs)
            embeddings.append(num_embs)

        # Concatenate embeddings
        transformer_inputs = keras.ops.concatenate(embeddings, axis=2)
        importances = []

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            if self.explainable:
                transformer_inputs, att_weights = block(transformer_inputs)
                att = att_weights[:, :, :, 0, :, :]
                att = keras.ops.sum(att, axis=(1, 2, 3))
                importances.append(att)
            else:
                transformer_inputs = block(transformer_inputs)

        if self.explainable:
            importances = keras.ops.sum(keras.ops.stack(importances), axis=0) / (
                self.depth * self.heads
            )
            return transformer_inputs, importances
        else:
            return transformer_inputs

   

class FTTransformer(keras.Model):
    def __init__(
        self,
        out_dim: int,
        out_activation: str,
        feature_unique_counts: dict,
        categorical_features: List[str] = None,
        numerical_features: List[str] = None,
        seq_length: int = 1,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_embedding_type: str = "linear",
        bins_dict: dict = None,
        n_bins: int = None,
        explainable: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.out_activation = keras.activations.get(out_activation)  # Get activation function
        self.ln = keras.layers.LayerNormalization(epsilon=1e-6)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(embedding_dim * seq_length // 2, activation="relu")
        self.dense2 = keras.layers.Dense(embedding_dim * seq_length // 4, activation="relu")
        self.output_layer = keras.layers.Dense(out_dim, activation=self.out_activation, name="output")

        self.transformer_encoder = FTTransformerEncoder(
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

    def call(self, inputs):
        x = self.transformer_encoder(inputs)
        layer_norm_cls = self.ln(x[:, :, 0, :])
        layer_norm_cls = self.flatten(layer_norm_cls)
        layer_norm_cls = self.dense1(layer_norm_cls)
        layer_norm_cls = self.dense2(layer_norm_cls)
        output = self.output_layer(layer_norm_cls)

        if self.transformer_encoder.explainable:
            return output, self.transformer_encoder.expl
        else:
            return output