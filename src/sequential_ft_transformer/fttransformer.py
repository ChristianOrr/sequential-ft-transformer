import keras
from sequential_ft_transformer.transformer import TransformerBlock
from sequential_ft_transformer.embeddings import (
    CatEmbeddingLayer,
    NumericEmbeddingLayer,
)
from typing import List, Dict, Optional


@keras.saving.register_keras_serializable(package="TransformerLayers")
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

    def get_config(self):
        config = {
            "seq_length": self.seq_length,
            "embedding_dim": self.embedding_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="TransformerLayers")
class FTTransformerEncoder(keras.layers.Layer):
    def __init__(self,
                categorical_features: List[str] = None,
                numerical_features: List[str] = None,
                feature_unique_counts: Dict[str, int] = None,
                seq_length: int = 1,
                embedding_dim: int = 16,
                depth: int = 4,
                heads: int = 8,
                attn_dropout: float = 0.2,
                ff_dropout: float = 0.2,
                numerical_embedding_type: str = "linear",
                bins_dict: Optional[Dict[str, List[float]]] = None,
                n_bins: Optional[int] = None,
                explainable: bool = False,
                post_norm: bool = False,
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
        self.post_norm = post_norm

    def build(self, input_shape):
        self.cls_weights_layer = CLSWeightsLayer(self.seq_length, self.embedding_dim)
        if self.categorical_features:
            self.cat_layer = CatEmbeddingLayer(
                feature_unique_counts=self.feature_unique_counts, emb_dim=self.embedding_dim
            )
        if self.numerical_features:
            self.numeric_layer = NumericEmbeddingLayer(
                feature_names=self.numerical_features,
                seq_length=self.seq_length,
                emb_dim=self.embedding_dim,
                emb_type=self.numerical_embedding_type,
                bins_dict=self.bins_dict,
                n_bins=self.n_bins,
            )
        self.transformer_blocks = [
            TransformerBlock(
                self.embedding_dim,
                self.heads,
                self.embedding_dim,
                attn_dropout=self.attn_dropout,
                ff_dropout=self.ff_dropout,
                explainable=self.explainable,
                post_norm=self.post_norm,
            )
            for _ in range(self.depth)
        ]

    def call(self, numeric_inputs=None, cat_inputs=None):
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

    def get_config(self):
        config = {
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "feature_unique_counts": self.feature_unique_counts,
            "seq_length": self.seq_length,
            "embedding_dim": self.embedding_dim,
            "depth": self.depth,
            "heads": self.heads,
            "attn_dropout": self.attn_dropout,
            "ff_dropout": self.ff_dropout,
            "numerical_embedding_type": self.numerical_embedding_type,
            "bins_dict": self.bins_dict,
            "n_bins": self.n_bins,
            "explainable": self.explainable,
            "post_norm": self.post_norm,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
   

@keras.saving.register_keras_serializable(package="TransformerLayers")
class FTTransformer(keras.Model):
    def __init__(
        self,
        out_dim: int,
        out_activation: str,
        feature_unique_counts: Dict[str, int] = None,
        categorical_features: List[str] = None,
        numerical_features: List[str] = None,
        seq_length: int = 1,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_embedding_type: str = "linear",
        bins_dict: Optional[Dict[str, List[float]]] = None,
        n_bins: Optional[int] = None,
        explainable: bool = False,
        dtype: str = None, # The saved model wont load without this argument
        **kwargs,
    ):
        self.out_dim = out_dim
        self.out_activation = keras.activations.get(out_activation)
        self.feature_unique_counts = feature_unique_counts
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
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

        # Initialize all the layers
        self.ln = keras.layers.LayerNormalization(epsilon=1e-6)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(self.embedding_dim * self.seq_length // 2, activation="relu")
        self.dense2 = keras.layers.Dense(self.embedding_dim * self.seq_length // 4, activation="relu")
        self.output_layer = keras.layers.Dense(self.out_dim, activation=self.out_activation, name="output")
        self.transformer_encoder = FTTransformerEncoder(
            categorical_features=self.categorical_features,
            numerical_features=self.numerical_features,
            feature_unique_counts=self.feature_unique_counts,
            seq_length=self.seq_length,
            embedding_dim=self.embedding_dim,
            depth=self.depth,
            heads=self.heads,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            numerical_embedding_type=self.numerical_embedding_type,
            bins_dict=self.bins_dict,
            n_bins=self.n_bins,
            explainable=self.explainable,
        )
        # Construct the network to allow for saving and loading the model
        inputs_layer = dict()
        empty_cat: bool = categorical_features is None or len(categorical_features) == 0
        empty_numeric: bool = numerical_features is None or len(numerical_features) == 0
        if not empty_cat:
            cat_inputs = keras.layers.Input(shape=(self.seq_length, len(self.categorical_features)))
            inputs_layer.update({"cat_inputs": cat_inputs})
        if not empty_numeric:
            numeric_inputs = keras.layers.Input(shape=(self.seq_length, len(self.numerical_features)))
            inputs_layer.update({"numeric_inputs": numeric_inputs})

        outputs_layer = self.call(inputs_layer)

        super(FTTransformer, self).__init__(inputs=inputs_layer,
                                        outputs=outputs_layer,
                                        **kwargs)
    
    def call(self, inputs):
        # numeric and/or cat inputs can be provided
        numeric_inputs = inputs.get("numeric_inputs")
        cat_inputs = inputs.get("cat_inputs")
        x = self.transformer_encoder(numeric_inputs=numeric_inputs, cat_inputs=cat_inputs)

        layer_norm_cls = self.ln(x[:, :, 0, :])
        layer_norm_cls = self.flatten(layer_norm_cls)
        layer_norm_cls = self.dense1(layer_norm_cls)
        layer_norm_cls = self.dense2(layer_norm_cls)
        output = self.output_layer(layer_norm_cls)

        if self.transformer_encoder.explainable:
            return output, self.transformer_encoder.expl
        else:
            return output
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_dim": self.out_dim,
                "out_activation": self.out_activation,
                "feature_unique_counts": self.feature_unique_counts,
                "categorical_features": self.categorical_features,
                "numerical_features": self.numerical_features,
                "seq_length": self.seq_length,
                "embedding_dim": self.embedding_dim,
                "depth": self.depth,
                "heads": self.heads,
                "attn_dropout": self.attn_dropout,
                "ff_dropout": self.ff_dropout,
                "numerical_embedding_type": self.numerical_embedding_type,
                "bins_dict": self.bins_dict,
                "n_bins": self.n_bins,
                "explainable": self.explainable,
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)