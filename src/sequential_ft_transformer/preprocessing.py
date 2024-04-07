import os
import urllib.request
import numpy as np
import pandas as pd
import tensorflow as tf
from ucimlrepo import fetch_ucirepo 

def pad_df(
    df: pd.DataFrame, 
    seq_length: int, 
    seq_col_name: str
):
    """
    Pads a DataFrame with sequential data to a specified sequence length.

    Args:
        df: The DataFrame to pad.
        seq_length: The desired sequence length.
        seq_col_name: The name of the column representing the sequence ID.

    Returns:
        The padded DataFrame.
    """
    grouped_df = df.groupby(seq_col_name)

    def g(df):
        num_df_rows = len(df)
        df = df.reset_index(drop=True)

        if num_df_rows > seq_length:
            # first rows
            # df = df[:seq_length]
            # last rows
            df = df[-seq_length:]
        elif num_df_rows < seq_length:
            num_padded_rows = seq_length - num_df_rows
            padded_rows = pd.DataFrame([df.iloc[-1]] * num_padded_rows)
            df = pd.concat([df, padded_rows], ignore_index=True)

        return df

    padded_df = grouped_df.apply(g)
    return padded_df.reset_index(drop=True)


def sq_df_to_dataset(
    input_df: pd.DataFrame, 
    seq_length: int,
    seq_col_name: str = None, 
    target_df: pd.DataFrame = None,
    target: str = None,
    categorical_features: list = None,
    numerical_features: list = None,  
    shuffle: bool = True,
    batch_size: int = 512,
):
    """
    Converts a DataFrame with sequential data to a TensorFlow dataset.
    Designed for the sequential FT-Transformer.

    Args:
        input_df: The input DataFrame.
        seq_length: The sequence length.
        seq_col_name: The name of the column representing the sequence ID (optional, for sequences > 1).
        target_df: The target DataFrame (optional).
        target: The target column name (optional).
        categorical_features: A list of categorical feature names (optional).
        numerical_features: A list of numerical feature names (optional).
        shuffle: Whether to shuffle the dataset (default: True).
        batch_size: The batch size for the dataset (default: 512).

    Returns:
        A TensorFlow dataset.
    """    

    input_df = input_df.copy()
    dataset = {}

    empty_cat: bool = categorical_features is None or len(categorical_features) == 0
    empty_numeric: bool = numerical_features is None or len(numerical_features) == 0

    if empty_cat and empty_numeric:
        raise ValueError("Both categorical and numerical features are missing. At least one is needed")
    if seq_length > 1 and seq_col_name is None:
        raise ValueError("The sequential column name is required when sequence length is greater then 1.")
    if not empty_cat:
        if seq_length > 1:
            cat_padded = pad_df(input_df[[seq_col_name] + categorical_features], seq_length, seq_col_name)
            cat_padded = cat_padded.drop(seq_col_name, axis=1)
            cat_array = cat_padded.to_numpy().reshape(-1, seq_length, len(categorical_features))
        else:
            cat_array = input_df[categorical_features].to_numpy()
            cat_array = cat_array.reshape(-1, seq_length, len(categorical_features))
        cat_array = cat_array.astype("int64")
        dataset["cat_inputs"] = cat_array
    if not empty_numeric:
        if seq_length > 1:
            numeric_padded = pad_df(input_df[[seq_col_name] + numerical_features], seq_length, seq_col_name)
            numeric_padded = numeric_padded.drop(seq_col_name, axis=1)
            numeric_array = numeric_padded.to_numpy().reshape(-1, seq_length, len(numerical_features))
        else:
            numeric_array = input_df[numerical_features].to_numpy()
            numeric_array = numeric_array.reshape(-1, seq_length, len(numerical_features))
        numeric_array = numeric_array.astype("float32")
        dataset["numeric_inputs"] = numeric_array

        
    if target is not None and target_df is not None:
        target_df = target_df.copy()
        labels = target_df.pop(target)
        dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(target_df))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


def download_data(
    url: str, 
    data_folder: str, 
    filename: str
):
    """
    Downloads a file from a URL to a specified folder.

    Args:
        url: The URL of the file to download.
        data_folder: The folder to download the file to.
        filename: The name to save the file as.
    """    

    filepath = os.path.join(data_folder, filename)

    # Create the data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)

    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename} to {data_folder}")
    else:
        print(f"{filename} already exists in {data_folder}")


def download_wine_dataset(
    data_folder: str = "../data", 
    filename: str = "wine_quality"
):  

    inputs_filepath = f"{os.path.join(data_folder, filename)}_inputs.csv"
    labels_filepath = f"{os.path.join(data_folder, filename)}_labels.csv"

    # Create the data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)

    if not os.path.exists(inputs_filepath) or not os.path.exists(labels_filepath):
        wine_quality = fetch_ucirepo(id=186)
        x = wine_quality["data"]["features"]
        y = wine_quality["data"]["targets"]
        x.to_csv(inputs_filepath, index=False, header=True)
        y.to_csv(labels_filepath, index=False, header=True)
        print(f"Downloaded {filename} to {data_folder}")
    else:
        print(f"{filename} already exists in {data_folder}")