
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "yfinance",
    "streamlit",
    "tensorflow",
    "keras"
]

# Install each package
for package in packages:
    install(package)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import streamlit as st

def get_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

def preprocess_data(df):
    df = df[['Close']]
    df = df.fillna(method='ffill')
    return df

def create_train_test_data(data, training_data_len):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], data[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, scaler

def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(data, train, valid, predictions):
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

def main():
    st.title('Stock Price Prediction App')

    st.sidebar.header('User Input Parameters')
    start = st.sidebar.text_input("Start Date", "2020-01-01")
    end = st.sidebar.text_input("End Date", "2023-01-01")
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL")

    df = get_data(symbol, start, end)
    data = preprocess_data(df)

    st.subheader(f'Closing Price\n')
    st.line_chart(data['Close'])

    training_data_len = int(np.ceil(len(data) * 0.8))
    x_train, y_train, x_test, y_test, scaler = create_train_test_data(data.values, training_data_len)

    model = build_model()
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    st.subheader('Predicted Closing Price\n')
    st.line_chart(valid[['Close', 'Predictions']])

if __name__ == '__main__':
    main()