import os
import urllib.request
import numpy as np
import pandas as pd
import tensorflow as tf


def pad_df(df, seq_length, seq_col_name):
  """Pads a dataframe so that all sequences have the same length.

  Args:
    df: A pandas dataframe with an 'id' column.
    seq_length: The desired length of each sequence.

  Returns:
    A padded dataframe where all sequences have the same length.
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
    input_df = input_df.copy()
    dataset = {}

    empty_cat: bool = categorical_features is None
    empty_numeric: bool = numerical_features is None

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


def df_to_dataset(
    dataframe: pd.DataFrame,  
    target: str = None,
    categorical_features: list = None,
    numerical_features: list = None,  
    shuffle: bool = True,
    batch_size: int = 512,
):
    df = dataframe.copy()
    dataset = {}

    empty_cat: bool = categorical_features is None
    empty_numeric: bool = numerical_features is None

    if empty_cat and empty_numeric:
        raise ValueError("Both categorical and numerical features are missing. At least one is needed")
    if not empty_cat:
        dataset["cat_inputs"] = df[categorical_features].to_numpy()
    if not empty_numeric:
        dataset["numeric_inputs"] = df[numerical_features].to_numpy()


    if target:
        labels = df.pop(target)
        dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


def download_data(url, data_folder, filename):
    """Downloads a file from a URL and saves it to a specified folder.

    Args:
        url (str): The URL of the file to download.
        data_folder (str): The path to the folder where the file should be saved.
        filename (str): The name to use for the downloaded file.
    """

    filepath = os.path.join(data_folder, filename)

    # Create the data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)

    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename} to {data_folder}")
    else:
        print(f"{filename} already exists in {data_folder}")