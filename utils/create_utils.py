import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import time
from termcolor import colored

def create_model(sequence_length, n_features, units=0, cell=LSTM, n_layers=0, 
dropout=0.0, loss="", optimizer="", bidirectional=False, activation=""):
    print(colored('starting model creation...', 'cyan'))
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True))) # no new input to provide, so return here
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss, metrics=[loss], optimizer=optimizer)
    model_name = f'{sequence_length}-{n_features}-{units}-{cell.__name__}-{n_layers}-{dropout}-{loss}-{optimizer}-{bidirectional}-{activation}-{time.time()}'
    print(colored(f'model creation complete: {model_name}', 'green'))
    return (model, model_name)