import yfinance as yf
import pandas as pd
import pendulum
import streamlit as st
import time
import schedule
from lstm import Lstm


def save_data_to_file(data):
    try:
        data.to_csv('stock_data.csv')
        print("Dane zostały zapisane do pliku:", 'stock_data.csv')
    except Exception as e:
        print("Wystąpił błąd podczas zapisywania danych do pliku:", str(e))


class DataManager:
    def __init__(self):
        today = pendulum.now()
        self.start_date = today.start_of('year').to_date_string()
        self.end_date = today.to_date_string()

        st.title("Stock trend prediction")
        user_input = st.text_input('Enter stock ticker', 'CL=F')

        # Define the ticker symbol for the stock
        self.ticker_symbol = user_input

    def fetch_stock_data(self):
        try:
            data = yf.download(self.ticker_symbol, start=self.start_date, end=self.end_date)
            return data
        except Exception as e:
            print("Wystąpił błąd podczas pobierania danych giełdowych:", str(e))
            return None

    def create_plot(self, stock_data, predicted_stock_price_with_actual, days):
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter, DayLocator

        # Plot the historical stock prices
        fig = plt.figure(figsize=(16, 10))
        plt.plot(stock_data.index, stock_data['Close'], label='Actual')
        plt.title(f"{self.ticker_symbol} Stock Price")
        plt.xlabel("Data")
        plt.ylabel("Cena")

        # Set the x-axis tick locator and formatter
        plt.gca().xaxis.set_major_locator(DayLocator(interval=7))
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

        #  Plot the predicted stock prices with the last actual price at the beginning
        predicted_dates = pd.bdate_range(start=self.end_date, periods=len(predicted_stock_price_with_actual))
        plt.plot(predicted_dates[:days+1], predicted_stock_price_with_actual[:days+1], label='Predicted')

        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

        st.subheader("Closing Price vs Date chart")
        st.pyplot(fig)

    def update_stock_data(self):
        # Download the historical stock data using yfinance
        stock_data = self.fetch_stock_data()

        if stock_data is None:
            # local file
            stock_data = pd.read_csv('stock_data.csv')
        else:
            # save to csv
            save_data_to_file(stock_data)

        #Describing data
        st.subheader('Data from 01-01-2023 to next 5 days')
        st.write(stock_data.describe())

        lstm = Lstm(stock_data)

        model = lstm.train_model()
        X_test, y_test, scaler = lstm.test_model()
        predicted_stock_price_with_actual, days = lstm.predict_price(X_test,y_test,model,scaler, self.end_date)


        self.create_plot(stock_data, predicted_stock_price_with_actual, days)

        print("Dane giełdowe zostały zaktualizowane.")

    def sched_local(self):
        # Ustawienie harmonogramu aktualizacji danych na godzinę 10:00
        schedule.every().day.at("10:00").do(self.update_stock_data)

        # Główna pętla programu
        while True:
            schedule.run_pending()
            time.sleep(1)