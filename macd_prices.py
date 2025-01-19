import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten

def load_and_display_macd(ticker1, ticker2, user_inputs):
    """
    Fetches historical data, calculates MACD and moving averages based on user input,
    trains LSTM-CNN models (for each ticker), and displays charts and predictions.

    Args:
        ticker1 (str): First ticker symbol.
        ticker2 (str): Second ticker symbol.
        user_inputs (dict): Dictionary containing user-specified parameters
            - price_column (str): "High", "Low", "Close", or "Open" (default: "Close")
            - ema_fast (int): Span for fast EMA (default: 12)
            - ema_slow (int): Span for slow EMA (default: 26)
            - signal_span (int): Span for signal EMA (default: 9)
    """

    try:
        data1 = yf.download(ticker1, period="3mo", interval="1h")
        data2 = yf.download(ticker2, period="3mo", interval="1h")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Get user-specified parameters (with defaults)
    price_column = user_inputs.get("price_column", "Close")
    ema_fast = user_inputs.get("ema_fast", 12)
    ema_slow = user_inputs.get("ema_slow", 26)
    signal_span = user_inputs.get("signal_span", 9)

    # Calculate MACD and Signal based on user input
    data1['ema_fast'] = data1[price_column].ewm(span=ema_fast, adjust=False).mean()
    data1['ema_slow'] = data1[price_column].ewm(span=ema_slow, adjust=False).mean()
    data1['MACD'] = data1['ema_fast'] - data1['ema_slow']
    data1['Signal'] = data1['MACD'].ewm(span=signal_span, adjust=False).mean()
    data1['Histogram'] = data1['MACD'] - data1['Signal']

    data2['ema_fast'] = data2[price_column].ewm(span=ema_fast, adjust=False).mean()
    data2['ema_slow'] = data2[price_column].ewm(span=ema_slow, adjust=False).mean()
    data2['MACD'] = data2['ema_fast'] - data2['ema_slow']
    data2['Signal'] = data2['MACD'].ewm(span=signal_span, adjust=False).mean()
    data2['Histogram'] = data1['MACD'] - data2['Signal']

    # Create candlestick charts with EMAs
    fig1_candlestick = go.Figure()
    fig1_candlestick.add_trace(go.Candlestick(x=data1.index,
                                             open=data1['Open'],
                                             high=data1['High'],
                                             low=data1['Low'],
                                             close=data1['Close'],
                                             name=f'{ticker1} Candlestick'))
    fig1_candlestick.add_trace(go.Scatter(x=data1.index, y=data1['ema_fast'], name=f'{ticker1} {ema_fast}-EMA', line=dict(color='red')))
    fig1_candlestick.add_trace(go.Scatter(x=data1.index, y=data1['ema_slow'], name=f'{ticker1} {ema_slow}-EMA', line=dict(color='blue')))
    fig1_candlestick.update_layout(title=f'{ticker1} Candlestick Chart with EMAs')

    fig2_candlestick = go.Figure()
    fig2_candlestick.add_trace(go.Candlestick(x=data2.index,
                                             open=data2['Open'],
                                             high=data2['High'],
                                             low=data2['Low'],
                                             close=data2['Close'],
                                             name=f'{ticker2} Candlestick'))
    fig2_candlestick.add_trace(go.Scatter(x=data2.index, y=data2['ema_fast'], name=f'{ticker2} {ema_fast}-EMA', line=dict(color='red')))
    fig2_candlestick.add_trace(go.Scatter(x=data2.index, y=data2['ema_slow'], name=f'{ticker2} {ema_slow}-EMA', line=dict(color='blue')))
    fig2_candlestick.update_layout(title=f'{ticker2} Candlestick Chart with EMAs')

    # Create line chart for MACD and Signal
    fig1_macd = go.Figure()
    fig1_macd.add_trace(go.Histogram(x=data1['Histogram'], name=f'{ticker1} MACD Histogram', opacity=0.5))
    fig1_macd.add_trace(go.Scatter(x=data1.index, y=data1['MACD'], name=f'{ticker1} MACD', line=dict(color='red')))
    fig1_macd.add_trace(go.Scatter(x=data1.index, y=data1['Signal'], name=f'{ticker1} Signal', line=dict(color='green')))
    fig1_macd.update_layout(title=f'{ticker1} MACD and Signal & Histogram')

    fig2_macd = go.Figure()
    fig2_macd.add_trace(go.Histogram(x=data2['Histogram'], name=f'{ticker2} MACD Histogram', opacity=0.5))
    fig2_macd.add_trace(go.Scatter(x=data2.index, y=data2['MACD'], name=f'{ticker2} MACD', line=dict(color='red')))
    fig2_macd.add_trace(go.Scatter(x=data2.index, y=data2['Signal'], name=f'{ticker2} Signal', line=dict(color='green')))
    fig2_macd.update_layout(title=f'{ticker2} MACD and Signal & Histogram')

    # Display charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1_candlestick)
        st.plotly_chart(fig1_macd)
    with col2:
        st.plotly_chart(fig2_candlestick)
        st.plotly_chart(fig2_macd)

    # Create features and target for training
    features_1 = data1[['ema_fast', 'ema_slow', 'MACD', 'Signal']]
    target_1 = data1[price_column]

    features_2 = data2[['ema_fast', 'ema_slow', 'MACD', 'Signal']]
    target_2 = data2[price_column]

    # Split data into training and testing sets (80/20 split)
    train_size_1 = int(len(data1) * 0.8)
    train_data_1, test_data_1 = features_1.iloc[:train_size_1], features_1.iloc[train_size_1:]
    train_target_1, test_target_1 = target_1.iloc[:train_size_1], target_1.iloc[train_size_1:]

    train_size_2 = int(len(data2) * 0.8)
    train_data_2, test_data_2 = features_2.iloc[:train_size_2], features_2.iloc[train_size_2:]
    train_target_2, test_target_2 = target_2.iloc[:train_size_2], target_2.iloc[train_size_2:]

    # Standardize data (using StandardScaler)
    scaler = StandardScaler()
    scaled_train_data_1 = scaler.fit_transform(train_data_1)
    scaled_test_data_1 = scaler.transform(test_data_1)

    scaled_train_data_2 = scaler.fit_transform(train_data_2)
    scaled_test_data_2 = scaler.transform(test_data_2)

    # Reshape data for LSTM-CNN input (samples, time steps, features)
    train_data_reshaped_1 = scaled_train_data_1.reshape(train_data_1.shape[0], train_data_1.shape[1], 1)
    test_data_reshaped_1 = scaled_test_data_1.reshape(test_data_1.shape[0], test_data_1.shape[1], 1)

    train_data_reshaped_2 = scaled_train_data_2.reshape(train_data_2.shape[0], train_data_2.shape[1], 1)
    test_data_reshaped_2 = scaled_test_data_2.reshape(test_data_2.shape[0], test_data_2.shape[1], 1)

    # Create LSTM-CNN models
    model1 = Sequential()
    model1.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(train_data_reshaped_1.shape[1], 1)))
    model1.add(MaxPooling1D(pool_size=2))
    model1.add(LSTM(50, return_sequences=True))
    model1.add(LSTM(50))
    # Add a Flatten layer before the Dense layer
    model1.add(Flatten())
    model1.add(Dense(features_1.shape[1]))  # Use the number of features from training data
    model1.compile(loss='mse', optimizer='adam')

    model2 = Sequential()
    model2.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(train_data_reshaped_2.shape[1], 1)))
    model2.add(MaxPooling1D(pool_size=2))
    model2.add(LSTM(50, return_sequences=True))
    model2.add(LSTM(50))
    model2.add(Flatten())
    model2.add(Dense(features_2.shape[1]))  # Use the number of features from training data
    model2.compile(loss='mse', optimizer='adam')

    # Train the models
    model1.fit(train_data_reshaped_1, train_target_1, epochs=50, batch_size=32, verbose=1)
    model2.fit(train_data_reshaped_2, train_target_2, epochs=50, batch_size=32, verbose=1)

    # Make predictions
    predictions_1 = model1.predict(test_data_reshaped_1)
    predictions_2 = model2.predict(test_data_reshaped_2)

    # Inverse transform predictions
    predictions_1 = scaler.inverse_transform(predictions_1)
    predictions_2 = scaler.inverse_transform(predictions_2)

    # Append the predicted prices to the lists
    last_prediction_1 = predictions_1[-1].reshape(1, -1)
    predictions_1 = np.concatenate((predictions_1, last_prediction_1), axis=0) 

    last_prediction_2 = predictions_2[-1].reshape(1, -1)
    predictions_2 = np.concatenate((predictions_2, last_prediction_2), axis=0) 

    # Update data for the next day's prediction
    # Note: This part might need adjustment based on your specific data handling and prediction logic
    # Here, we simply append the last prediction to the existing data 
    # In a real-world scenario, you might need to shift the data and handle windows more carefully

    # Create a DataFrame for predicted prices
    prediction_df = pd.DataFrame({'Predicted Price (' + ticker1 + ')': predictions_1[:, 0],  # Assuming single-step price prediction
                                 'Predicted Price (' + ticker2 + ')': predictions_2[:, 0]},
                                 index=pd.date_range(start=data1.index[-1] + pd.Timedelta(days=1), 
                                                    periods=len(predictions_1)))

    # Display the predicted prices
    st.write("Predicted Prices for the Next Day:")
    st.dataframe(prediction_df)

    # # Display predictions
    # st.dataframe(predictions_1)
    # st.dataframe(predictions_2)
    # st.write("Predictions for Next 3 Days:")
    # st.write(f"{ticker1}: {predictions_1[-3:]}")
    # st.write(f"{ticker2}: {predictions_2[-3:]}")