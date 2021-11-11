"""
Comment out imports until needed. TF slows down the program drastically
"""
# import tensorflow as tf
import numpy as np
import pandas as pd
import random, json

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque
from termcolor import colored

with open("utils/ticker_to_csv.json",'r') as f:
    ticker_to_csv = json.load(f)

SEED_VAL = 123

np.random.seed(SEED_VAL)
# tf.random.set_seed(SEED_VAL)
random.seed(SEED_VAL)

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def load_data(ticker="", n_steps=None, scale=None, shuffle=None, lookup_step=None, split_by_date=None, test_size=None, feature_columns=[]):
    print(colored('starting data load...', 'magenta'))

    if isinstance(ticker, str) and ticker_to_csv.get(ticker) is not None:
        file_name = 'cryptoData/' + ticker_to_csv.get(ticker)
        df = pd.read_csv(file_name)
        # print(df.head())
    elif isinstance(ticker, str) and ticker_to_csv.get(ticker) is None:
        print("please use a supported ticker symbol -", *ticker_to_csv.keys())
        return None
    else:
        print("ticker can only be of type str")
        return None

    result = {}

    result["df"] = df.copy()

    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    if "Date" not in df.columns:
        df["Date"] = df.index

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # future price is set to the lookup_step amount of days ahead from current
    df['future'] = df['Close'].shift(-lookup_step)

    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["Date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence

    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    dates = result["X_test"][:, -1, -1]

    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].reindex([dates], axis=1) # swapped .loc with .reindex, source: https://stackoverflow.com/questions/65422225/how-to-solve-keyerrorfnone-of-key-are-in-the-axis-name-in-this-case
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    print(colored('test data creation complete', 'green'))
    return result
