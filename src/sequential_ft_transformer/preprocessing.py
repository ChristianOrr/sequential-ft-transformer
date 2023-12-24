import os
import urllib.request
import numpy as np
import pandas as pd
import tensorflow as tf


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


# def df_to_dataset(
#     dataframe: pd.DataFrame,
#     target: str = None,
#     shuffle: bool = True,
#     batch_size: int = 512,
# ):
#     df = dataframe.copy()
#     if target:
#         labels = df.pop(target)
#         dataset = {}
#         for key, value in df.items():
#             dataset[key] = value.to_numpy()[:, tf.newaxis]

#         dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
#     else:
#         dataset = {}
#         for key, value in df.items():
#             dataset[key] = value.to_numpy()[:, tf.newaxis]

#         dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))

#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=len(dataframe))
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(batch_size)
#     return dataset


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