import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

class Lstm:
    def __init__(self, data, column_name='Close'):
        self.stock_data = data
        self.data = self.stock_data[[column_name]].values
        self.column_name = column_name

    def prepare_data(self, train_size_value = 0.8 ,time_steps=60):
        # Prepare the data for training
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)

        # Split the data into training and testing sets
        train_size = int(len(scaled_data) * train_size_value)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Create input sequences for training
        X_train = []
        y_train = []
        for i in range(time_steps, len(train_data)):
            X_train.append(train_data[i - time_steps:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Reshape the input sequences for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        return scaler, X_train, y_train, test_data

    def train_model(self,time_steps=60, optimizer='adam', loss='mean_squared_error', epochs=50, batch_size=32):

        scaler, X_train, y_train, test_data = self.prepare_data()

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(time_steps, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=128))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile and train the model
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        return model

    def test_model(self, time_steps=60):
        scaler, _, _, test_data = self.prepare_data()
        # Prepare the data for testing
        inputs = self.data[len(self.data) - len(test_data) - time_steps:]
        inputs = scaler.transform(inputs)

        # Create input sequences for testing
        X_test = []
        y_test = []
        for i in range(time_steps, len(inputs)):
            X_test.append(inputs[i - time_steps:i, 0])
            y_test.append(inputs[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)

        return X_test, y_test, scaler

    def predict_price(self, X_test, y_test, model, scaler, end_date, days=5):
        # Reshape the input sequences for LSTM
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Make predictions on the test data
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        # Get the last point from the actual stock data
        last_actual_price = self.stock_data[self.column_name].iloc[-1]

        # Insert the last actual price at the beginning of the predicted prices
        predicted_stock_price_with_actual = np.insert(predicted_stock_price, 0, last_actual_price)

        mse = mean_squared_error(y_test, predicted_stock_price)
        print("Mean Squared Error (MSE):", mse)
        rmse = np.sqrt(np.mean(((predicted_stock_price - y_test) ** 2)))
        print(f"RMSE: {rmse}")

        # Print the predicted stock prices for the next 5 days
        next_five_days = predicted_stock_price[-days:]
        next_five_days_dates = pd.bdate_range(start=end_date, periods=days)
        for date, price in zip(next_five_days_dates, next_five_days):
            print(f"{date.date()}: {price[0]}")

        return predicted_stock_price_with_actual, days
