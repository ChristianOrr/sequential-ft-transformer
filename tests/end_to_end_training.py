import os
os.environ["KERAS_BACKEND"] = "jax"
import pandas as pd
import numpy as np
import keras
from keras.optimizers import AdamW
from keras.losses import MeanSquaredError
from keras.metrics import R2Score
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sequential_ft_transformer.fttransformer import FTTransformer
from sequential_ft_transformer.preprocessing import sq_df_to_dataset, download_wine_dataset
from typing import List, Dict, Optional


def test_pandas_df(
        df: pd.DataFrame, 
        columns: List[str] = None,
    ):
    """
    Performs basic tests on a pandas DataFrame

    Args:
        df: The pandas DataFrame to test
        columns: Desired column names
      
    Returns:
        None
    """
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert df.shape[0] > 0, "DataFrame is empty"
    assert len(df.columns) > 0, "DataFrame has no columns"

    if columns is not None:
        df_cols_list = sorted(list(df.columns))
        desired_cols_list = sorted(columns)
        assert len(df_cols_list) == len(desired_cols_list), "The number of columns doesnt match"
        for df_col_name, required_col_name in zip(df_cols_list, desired_cols_list):
            assert df_col_name == required_col_name, f"Column names do not match, df name: {df_col_name}, required name: {required_col_name}"
        assert df_cols_list == desired_cols_list, "The column lists do not match"



def train_model(
    data_folder="../data/",
    batch_size=100,
    seq_length=1,
    numeric_features=[],
    cat_features=[],
    label='quality',
    explanations=False,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_epochs=1,
):

    download_wine_dataset(data_folder=data_folder)

    X = pd.read_csv(os.path.join(data_folder, "wine_quality_inputs.csv"))
    Y = pd.read_csv(os.path.join(data_folder, "wine_quality_labels.csv"))

    features = sorted(list(numeric_features) + list(cat_features))

    test_pandas_df(X, columns=features)
    test_pandas_df(Y, columns=[label])

    train_input, test_input, train_label, test_label = train_test_split(X, Y, test_size=0.1)


    train_dataset = sq_df_to_dataset(
        train_input,
        seq_length,
        target_df=train_label,
        target=label,
        numerical_features=numeric_features,
        batch_size=batch_size,
    )
    test_dataset = sq_df_to_dataset(
        test_input,
        seq_length,
        numerical_features=numeric_features,
        shuffle=False,
        batch_size=batch_size,
    )

    model = FTTransformer(
        out_dim=1,
        out_activation='linear',
        numerical_features=numeric_features,
        seq_length=seq_length,
        embedding_dim=8,
        depth=1,
        heads=2,
        numerical_embedding_type='linear',
        explainable=explanations,
    )

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss = MeanSquaredError()
    metrics = [R2Score()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    print("Start of training...")
    hist = model.fit(
        train_dataset,
        epochs=num_epochs,
    )

    assert model.built, "Model failed to build"

    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)

    print("Saving the model...")
    model.save(f"{models_dir}/model.keras")
    
    loaded_model = keras.models.load_model(f"{models_dir}/model.keras")

    model_preds = model.predict(test_dataset)
    loaded_model_preds = loaded_model.predict(test_dataset)

    np.testing.assert_allclose(
        model_preds, loaded_model_preds
    )