import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import (
    Dense,
    Flatten,
)
import math as m

from sequential_ft_transformer.transformer import TransformerBlock, transformer_block
from sequential_ft_transformer.embeddings import (
    cat_embedding,
    num_embedding,
    CEmbedding, 
    NEmbedding
)


def ft_transformer_encoder(
    numeric_inputs: layers.Input,
    cat_inputs: layers.Input,
    categorical_features: list,
    numerical_features: list,
    numerical_data: np.array,
    feature_unique_counts: list,
    task: str = None,
    seq_length: int = 1,
    embedding_dim: int = 32,
    depth: int = 4,
    heads: int = 8,
    attn_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    numerical_embedding_type: str = 'linear',
    numerical_bins: int = None,
    ple_tree_params: dict = {},
    explainable=False,       
):
    # Define the layers
    # CLS token
    w_init = tf.random_normal_initializer()
    if numeric_inputs is not None:
        batch_size = tf.shape(numeric_inputs)[0]
    elif cat_inputs is not None:
        batch_size = tf.shape(cat_inputs)[0]
    else:
        raise ValueError("Numeric and Categorical inputs are None.")

    shape = tf.stack([batch_size, seq_length, 1, embedding_dim], axis=0)
    cls_weights = w_init(shape)
    transformer_inputs = [cls_weights]


    # If categorical features, add to list
    if categorical_features is not None and cat_inputs is not None:
        cat_embs = cat_embedding(
            inputs=cat_inputs,
            feature_names=categorical_features,
            feature_unique_counts=feature_unique_counts,
            emb_dim =embedding_dim
        )
        transformer_inputs += [cat_embs]
    
    # If numerical features, add to list
    if numerical_features is not None and numeric_inputs is not None:
        num_embs = num_embedding(
            inputs=numeric_inputs,
            feature_names=numerical_features, 
            X=numerical_data, 
            y=None,
            task=task,
            seq_length=seq_length,
            emb_dim=embedding_dim, 
            emb_type=numerical_embedding_type, 
            n_bins=numerical_bins,
            tree_params=ple_tree_params
        )

        transformer_inputs += [num_embs]
    
    # Prepare for Transformer
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
        # Sum across the layers
        importances = tf.reduce_sum(tf.stack(importances), axis=0) / (
            depth * heads
        )
        return transformer_inputs, importances
    else:
        return transformer_inputs, None
        



def ft_transformer(
    out_dim: int,
    out_activation: str,
    numerical_data: np.array,
    feature_unique_counts: list,
    categorical_features: list = None,
    numerical_features: list = None,
    seq_length: int = 1,
    embedding_dim: int = 32,
    depth: int = 4,
    heads: int = 8,
    attn_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    numerical_embedding_type: str = None,
    explainable=False,      
):
    # mlp layers
    ln = tf.keras.layers.LayerNormalization()
    flatten = tf.keras.layers.Flatten()
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
        # numeric_input_shape = (len(numerical_features), )
        numeric_input_shape = (seq_length, len(numerical_features), )
        numeric_inputs = layers.Input(shape=numeric_input_shape, name="numeric_inputs")
        inputs_dict.update({"numeric_inputs": numeric_inputs})
    else:
        numeric_inputs = None

    if not empty_cat:
        # cat_input_shape = (len(categorical_features), )
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
        numerical_data=numerical_data,
        feature_unique_counts=feature_unique_counts,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        numerical_embedding_type=numerical_embedding_type,
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






# OLD CODE


class FTTransformerEncoder(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list,
        numerical_features: list,
        numerical_data: np.array,
        categorical_data: np.array,
        y: np.array = None,
        task: str = None,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_embedding_type: str = 'linear',
        numerical_bins: int = None,
        ple_tree_params: dict = {},
        explainable=False,
    ):
        """FTTransformer Encoder
        Args:
            categorical_features (list): names of categorical features
            numerical_features (list): names of numeric features
            categorical_lookup (dict): dictionary with categorical feature names as keys and adapted StringLookup layers as values
            out_dim (int): model output dimensions
            out_activation (str): model output activation
            embedding_dim (int, optional): embedding dimensions. Defaults to 32.
            depth (int, optional): number of transformer blocks. Defaults to 4.
            heads (int, optional): number of attention heads. Defaults to 8.
            attn_dropout (float, optional): dropout rate in transformer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in mlps. Defaults to 0.1.
            mlp_hidden_factors (list[int], optional): numbers by which we divide dimensionality. Defaults to [2, 4].
            numerical_embeddings (dict, optional): dictionary with numerical feature names as keys and adapted numerical embedding layers as values. Defaults to None.
            numerical_embedding_type (str, optional): name of the numerical embedding procedure. Defaults to linear.
            use_column_embedding (bool, optional): flag to use fixed column positional embeddings. Defaults to True.
            explainable (bool, optional): flag to output importances inferred from attention weights. Defaults to False.
        """

        super(FTTransformerEncoder, self).__init__()
        self.numerical = numerical_features
        self.categorical = categorical_features
        self.numerical_embedding_type = numerical_embedding_type
        self.embedding_dim = embedding_dim
        self.explainable = explainable
        self.depth = depth
        self.heads = heads
            
        # Two main embedding modules
        if len(self.numerical) > 0:
            self.numerical_embeddings = NEmbedding(
                feature_names=self.numerical, 
                X=numerical_data, 
                y=y,
                task=task,
                emb_dim=embedding_dim, 
                emb_type=numerical_embedding_type, 
                n_bins=numerical_bins,
                tree_params=ple_tree_params
            )
        if len(self.categorical) > 0:
            self.categorical_embeddings = CEmbedding(
                feature_names=self.categorical,
                X=categorical_data,
                emb_dim =embedding_dim
            )

        # Transformers
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(
                    embedding_dim,
                    heads,
                    embedding_dim,
                    att_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    explainable=self.explainable,
                    post_norm=False,  # FT-Transformer uses pre-norm
                )
            )
        self.flatten_transformer_output = Flatten()

        # CLS token
        w_init = tf.random_normal_initializer()
        self.cls_weights = tf.Variable(
            initial_value=w_init(shape=(1, embedding_dim), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        # Start with CLS token
        cls_tokens = tf.repeat(self.cls_weights, repeats=tf.shape(inputs[self.numerical[0]])[0], axis=0)
        cls_tokens = tf.expand_dims(cls_tokens, axis=1)
        transformer_inputs = [cls_tokens]
    
        # If categorical features, add to list
        if len(self.categorical) > 0:
            cat_input = []
            for c in self.categorical:
                cat_input.append(inputs[c])
            
            cat_input = tf.stack(cat_input, axis=1)[:, :, 0]
            cat_embs = self.categorical_embeddings(cat_input)
            transformer_inputs += [cat_embs]
        
        # If numerical features, add to list
        if len(self.numerical) > 0:
            num_input = []
            for n in self.numerical:
                num_input.append(inputs[n])
            num_input = tf.stack(num_input, axis=1)[:, :, 0]
            num_embs = self.numerical_embeddings(num_input)
            transformer_inputs += [num_embs]
        
        # Prepare for Transformer
        transformer_inputs = tf.concat(transformer_inputs, axis=1)
        importances = []
        
        # Pass through Transformer blocks
        for transformer in self.transformers:
            if self.explainable:
                transformer_inputs, att_weights = transformer(transformer_inputs)
                print(f"att_weights.shape: {att_weights.shape}")
                alt_weights = att_weights[:, :, 0, :]
                print(f"alt_weights.shape: {alt_weights.shape}")
                agg_att = tf.reduce_sum(alt_weights, axis=1)
                print(f"agg_att.shape: {agg_att.shape}")
                importances.append(agg_att)
            else:
                transformer_inputs = transformer(transformer_inputs)

        if self.explainable:
            # Sum across the layers
            importances = tf.reduce_sum(tf.stack(importances), axis=0) / (
                self.depth * self.heads
            )
            return transformer_inputs, importances
        else:
            return transformer_inputs


class FTTransformer(tf.keras.Model):
    def __init__(
        self,
        out_dim: int,
        out_activation: str,
        categorical_features: list = None,
        numerical_features: list = None,
        categorical_lookup: dict = None,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_embedding_type: str = None,
        numerical_embeddings: dict = None,
        explainable=False,
        encoder=None,
    ):
        super(FTTransformer, self).__init__()

        # Initialise encoder
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = FTTransformerEncoder(
                categorical_features = categorical_features,
                numerical_features = numerical_features,
                categorical_lookup = categorical_lookup,
                embedding_dim = embedding_dim,
                depth = depth,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                numerical_embedding_type = numerical_embedding_type,
                numerical_embeddings = numerical_embeddings,
                explainable = explainable,
            )

        # mlp layers
        self.ln = tf.keras.layers.LayerNormalization()
        self.final_ff = Dense(embedding_dim//2, activation='relu')
        self.output_layer = Dense(out_dim, activation=out_activation)
    
    def call(self, inputs):
        if self.encoder.explainable:
            x, expl = self.encoder(inputs)
        else:
            x = self.encoder(inputs)

        layer_norm_cls = self.ln(x[:, 0, :])
        layer_norm_cls = self.final_ff(layer_norm_cls)
        output = self.output_layer(layer_norm_cls)

        if self.encoder.explainable:
            # Explaianble models return two outputs
            return {"output": output, "importances": expl}
        else:
            return output