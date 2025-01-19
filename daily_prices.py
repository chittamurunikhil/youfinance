import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_and_display_prices(ticker1, ticker2):

    # Fetch historical data
    try:
        data1 = yf.download(ticker1, period="5y")
        data2 = yf.download(ticker2, period="5y")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Calculate daily change in closing price
    data1['Daily Change (' + ticker1 + ')'] = data1['Close'].diff()
    data2['Daily Change (' + ticker2 + ')'] = data2['Close'].diff()

    # ---- Chart of Daily Changes ----
    fig = go.Figure()

    # Create line traces for daily changes
    fig.add_trace(go.Scatter(
        x=data1.index,
        y=data1['Daily Change (' + ticker1 + ')'],
        name='Daily Change (' + ticker1 + ')',
        mode='lines',
        marker=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatter(
        x=data2.index,
        y=data2['Daily Change (' + ticker2 + ')'],
        name='Daily Change (' + ticker2 + ')',
        mode='lines',
        marker=dict(color='darkorange')
    ))

    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Daily Change',
        title='Daily Change in Closing Price for ' + ticker1 + ' and ' + ticker2
    )

    # Display the chart
    st.plotly_chart(fig)

    # Calculate 50-day and 100-day moving averages
    data1['50-Day MA (' + ticker1 + ')'] = data1['Close'].rolling(window=50).mean()
    data1['100-Day MA (' + ticker1 + ')'] = data1['Close'].rolling(window=100).mean()
    data2['50-Day MA (' + ticker2 + ')'] = data2['Close'].rolling(window=50).mean()
    data2['100-Day MA (' + ticker2 + ')'] = data2['Close'].rolling(window=100).mean()

    # ---- Candlestick Chart with Moving Averages ----
    fig_1 = go.Figure()

    # Add candlestick traces for each ticker
    fig_1.add_trace(go.Candlestick(
        x=data1.index,
        open=data1['Open'],
        high=data1['High'],
        low=data1['Low'],
        close=data1['Close'],
        name=ticker1
    ))
    fig_1.add_trace(go.Candlestick(
        x=data2.index,
        open=data2['Open'],
        high=data2['High'],
        low=data2['Low'],
        close=data2['Close'],
        name=ticker2
    ))

    # Add moving average lines for each ticker
    fig_1.add_trace(go.Scatter(
        x=data1.index,
        y=data1['50-Day MA (' + ticker1 + ')'],
        name='50-Day MA (' + ticker1 + ')',
        mode='lines',
        line=dict(color='blue', width=2)
    ))
    fig_1.add_trace(go.Scatter(
        x=data1.index,
        y=data1['100-Day MA (' + ticker1 + ')'],
        name='100-Day MA (' + ticker1 + ')',
        mode='lines',
        line=dict(color='green', width=2)
    ))
    fig_1.add_trace(go.Scatter(
        x=data2.index,
        y=data2['50-Day MA (' + ticker2 + ')'],
        name='50-Day MA (' + ticker2 + ')',
        mode='lines',
        line=dict(color='orange', width=2)
    ))
    fig_1.add_trace(go.Scatter(
        x=data2.index,
        y=data2['100-Day MA (' + ticker2 + ')'],
        name='100-Day MA (' + ticker2 + ')',
        mode='lines',
        line=dict(color='teal', width=2)
    ))

    # Customize the layout
    fig_1.update_layout(
        xaxis_title='Date',
        yaxis_title='Price'
    )

    # Display the chart
    st.plotly_chart(fig_1)

    # ---- Machine Learning Model for Future Price Prediction ----
    def create_model(x_train, y_train):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))  # Explicitly define input shape
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.01))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(x_train, y_train, epochs=5, batch_size=32 , verbose=1)
        return model

    # Hyperparameters (adjust as needed)
    look_back = 500  # Number of days to look back for prediction
    n_features = 1  # Using closing price as the feature

    # Prepare data for model (assuming closing price for prediction)
    scaler_1 = MinMaxScaler(feature_range=(0, 1))
    scaler_2 = MinMaxScaler(feature_range=(0, 1))
    scaled_data1 = scaler_1.fit_transform(data1[['Close']])
    scaled_data2 = scaler_2.fit_transform(data2[['Close']])

    x_train1, y_train1 = [], []
    x_train2, y_train2 = [], []

    for i in range(look_back, len(scaled_data1)):
        x_train1.append(scaled_data1[i-look_back:i, 0])
        y_train1.append(scaled_data1[i, 0])

    for i in range(look_back, len(scaled_data2)):
        x_train2.append(scaled_data2[i-look_back:i, 0])
        y_train2.append(scaled_data2[i, 0])

    x_train1, y_train1 = np.array(x_train1), np.array(y_train1)
    x_train2, y_train2 = np.array(x_train2), np.array(y_train2)

    # Reshape input data to 3D format (samples, timesteps, features)
    x_train1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], n_features))
    x_train2 = np.reshape(x_train2, (x_train2.shape[0], x_train2.shape[1], n_features))

    # Train the model (feel free to experiment with different architectures)
    model1 = create_model(x_train1, y_train1)
    model2 = create_model(x_train2, y_train2)

    # ---- User Interface for Prediction ----
    prediction_days = st.slider('Number of Days to Predict', 1, 30, 7)

    # Make predictions
    predicted_prices1 = []
    predicted_prices2 = []

    for i in range(prediction_days):
        # Get the last 'look_back' days of data for each ticker
        last_n_days1 = scaled_data1[-look_back:]
        last_n_days2 = scaled_data2[-look_back:]

        # Reshape input data to 3D format (samples, timesteps, features)
        x_test1 = np.array([last_n_days1]).reshape(1, look_back, n_features) 
        x_test2 = np.array([last_n_days2]).reshape(1, look_back, n_features)

        # Predict future prices
        predicted_price1 = model1.predict(x_test1)
        predicted_price2 = model2.predict(x_test2)

        # Inverse scale the predicted prices
        predicted_price1 = scaler_1.inverse_transform(predicted_price1)
        predicted_price2 = scaler_2.inverse_transform(predicted_price2)

        # Append the predicted prices to the lists
        predicted_prices1.append(predicted_price1.flatten()[0])
        predicted_prices2.append(predicted_price2.flatten()[0])

        # Update data for the next day's prediction
        scaled_data1 = np.append(scaled_data1[1:], predicted_price1.flatten())
        scaled_data2 = np.append(scaled_data2[1:], predicted_price2.flatten())

    # Create a DataFrame for predicted prices
    prediction_df = pd.DataFrame({'Predicted Price (' + ticker1 + ')': predicted_prices1,
                                  'Predicted Price (' + ticker2 + ')': predicted_prices2},
                                 index=pd.date_range(start=data1.index[-1] + pd.Timedelta(days=1),
                                                      periods=prediction_days))

    # Display the predicted prices
    st.write("Predicted Prices for the Next", prediction_days, "Days:")
    st.dataframe(prediction_df)

    # Get current prices
    try:
        current_price1 = yf.download(ticker1, period="1d")['Close'][0]
        current_price2 = yf.download(ticker2, period="1d")['Close'][0]
    except Exception as e:
        st.error(f"Error fetching current prices: {e}")
        return

    # Display current prices
    st.write(f"**Current Price ({ticker1}): ** {current_price1:.2f}")
    st.write(f"**Current Price ({ticker2}): ** {current_price2:.2f}")


    # Calculate prediction errors for each day
    prediction_df['Error (' + ticker1 + ')'] = prediction_df['Predicted Price (' + ticker1 + ')'] - current_price1
    prediction_df['Error (' + ticker2 + ')'] = prediction_df['Predicted Price (' + ticker2 + ')'] - current_price2 

    

    # Display the prediction errors along with predicted prices
    st.write("Predicted Prices and Errors for the Next", prediction_days, "Days:")
    st.dataframe(prediction_df)



