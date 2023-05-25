## Prognozowanie cen akcji
Utworzony projekt pozwala na prognozowanie cen akcji za pomocą modelu LSTM, używając języka Python. Model analizuje historyczne dane, a następnie na ich podstawie stara się przewidzieć ceny akcji na następne X dni.

## Wymagania
Aby uruchomić projekt, wymagane są następujące biblioteki:
1. yfinance
2. pandas
3. numpy
4. pendulum
5. streamlit
6. time
7. schedule
8. lstm
9. sklearn
10. keras

## Sposób działania
1. Program pobiera za pomocą biblioteki `yfinance` dane historyczne dotyczące wybranych akcji (domyślnie CL=F). Następnie dane te są zapisywane do pliku stock_data.csv.
2. W przypadku problemów z pobraniem danych, program wykorzysta dane zapisane w pliku.
3. Kolejnym krokiem jest dokonanie analizy danych i predykcja cen - modyfikując kod istnieje możliwość łatwej konfiguracji modelu oraz liczby dni, na jakie ma być dokonywana predykcja.
4. Program automatycznie dokonuje aktualizacji danych codziennie o godzinie 10:00.
5. Dzięki wykorzystaniu biblioteki `Streamlit`, aplikacja będzie wyświetlała wyniki w przeglądarce internetowej. W ten sposób można rówinież zmienić, dla jakich akcji program ma dokonać predykcji.

## Sposób uruchomienia
Aby uruchomić program, należy:
1. Przejść do katalogu z projektem (folder z plikiem main.py)
2. Uruchomić terminal
3. Wydać polecenie 
```shell
streamlit run main.py
```

## Autor
Artur Sulej
