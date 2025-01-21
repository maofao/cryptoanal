import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests
import datetime
import time

# Ваш API ключ для CryptoCompare (получите на https://min-api.cryptocompare.com/)
api_key = 'c32d179556514500fe771c410ab021ff743c722146ae35799bf557ab4e0c32d6'

# Запрашиваем данные для криптовалюты (например, BTC)
crypto_id = 'BTC'  # Используем символ криптовалюты, например, BTC для Bitcoin
currency = 'USD'  # В какой валюте запрашиваем цену (например, USD)

# URL для получения исторических данных (с параметрами запроса)
url = 'https://min-api.cryptocompare.com/data/v2/histominute'

# Параметры запроса (например, BTC/USD, интервал 1 минута)
params = {
    'fsym': crypto_id,
    'tsym': currency,
    'limit': 2000,  # Максимум 2000 данных за один запрос
    'toTs': int(datetime.datetime.now().timestamp()),  # Время до текущего момента
    'api_key': api_key
}

def fetch_data():
    # Получаем данные
    response = requests.get(url, params=params)
    data = response.json()

    # Преобразуем данные в DataFrame
    ohlc_data = data['Data']['Data']
    prices_df = pd.DataFrame(ohlc_data)

    prices_df['Date'] = pd.to_datetime(prices_df['time'], unit='s')
    prices_df.set_index('Date', inplace=True)


    prices_df = prices_df[['close']]

    # Преобразуем данные в числовой формат
    prices_df['close'] = pd.to_numeric(prices_df['close'])
    
    return prices_df

def create_dataset(data, time_step=60):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

# Построение улучшенной модели LSTM с Dropout 
def build_model(x_train):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Добавлен слой Dropout
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))  # Добавлен слой Dropout
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Основная функция, которая будет обновлять предсказания каждую минуту
def predict_price():
    # Загружаем последние данные
    prices_df = fetch_data()
    
    # Преобразуем данные
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices_df['close'].values.reshape(-1, 1))

    # Создание обучающих и тестовых данных
    time_step = 60
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Создание обучающих и тестовых наборов
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)

    # Форматирование данных для LSTM
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Строим модель
    model = build_model(x_train)

    # Обучаем модель
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)  # Быстро обучаем модель

    # Прогнозируем с использованием тестовых данных
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Отображаем предсказания
    plt.figure(figsize=(10, 6))
    plt.plot(prices_df.index[train_size:], prices_df['close'][train_size:], color='blue', label='Actual Price')
    plt.plot(prices_df.index[train_size + time_step + 1:], predictions, color='red', label='Predicted Price')
    plt.legend()
    plt.title(f'Price Prediction for {crypto_id}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.show()

# Функция для непрерывного обновления предсказаний каждую минуту
def continuous_prediction():
    while True:
        predict_price()
        time.sleep(60)  # Ждем 1 минуту

# Запускаем непрерывное обновление
continuous_prediction()
