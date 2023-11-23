import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt


# Đọc dữ liệu từ tệp CSV
def read_stock_data(filename):
    df = pd.read_csv(filename)
    return df
#12123


# Xử lý dữ liệu
def preprocess_data(df):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
def split_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data, test_data = data[:train_size], data[train_size:]
    return train_data, test_data


# Xây dựng dữ liệu đầu vào cho mô hình RNN
def create_sequences(data, sequence_length=10):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)


# Xây dựng mô hình RNN
def build_rnn_model(sequence_length):
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Huấn luyện mô hình
def train_model(model, train_data, epochs=50, batch_size=32):
    X_train = create_sequences(train_data)
    y_train = train_data[10:]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)


# Đánh giá mô hình
def evaluate_model(model, test_data, scaler, sequence_length=10):
    X_test = create_sequences(test_data, sequence_length)
    y_true = test_data[sequence_length:]
    y_pred = model.predict(X_test)
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return mse, y_true, y_pred


if __name__ == "__main__":
    # Tệp dữ liệu cho mỗi mã cổ phiếu
    stock_files = ["data/FPT.csv", "data/MSN.csv", "data/PNJ.csv", "data/VIC.csv"]

    for stock_file in stock_files:
        # Đọc dữ liệu từ tệp CSV
        df = read_stock_data(stock_file)

        # Xử lý dữ liệu
        data, scaler = preprocess_data(df)

        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        train_data, test_data = split_data(data)

        # Xây dựng và huấn luyện mô hình RNN
        sequence_length = 10  # Độ dài của chuỗi đầu vào
        model = build_rnn_model(sequence_length)
        train_model(model, train_data, epochs=50, batch_size=32)

        # Đánh giá mô hình và in ra kết quả
        mse, y_true, y_pred = evaluate_model(model, test_data, scaler, sequence_length)
        print(f"Stock: {stock_file}")
        print(f"Mean Squared Error (MSE): {mse}")

        # Vẽ biểu đồ kết quả
        plt.figure()
        plt.plot(y_true, label='Actual Price')
        plt.plot(y_pred, label='Predicted Price')
        plt.legend()
        plt.title(f"Stock Price Prediction for {stock_file}")
        plt.show()
        #adasdasasdf