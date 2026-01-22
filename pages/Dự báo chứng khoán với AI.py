from vnstock import Quote
import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai

# Input for stock symbol
st.markdown(
    "<h1 style='color: darkblue;'>📊 Dự báo chứng khoán Việt Nam dựa trên phân tích AI và Máy học</h1>",
    unsafe_allow_html=True
)
symbol = st.text_input("Nhập mã chứng khoán vào đây (ví dụ, VCB):", value="VCB")

# Date pickers for start and end dates
start_date = st.date_input("Lựa chọn ngày bắt đầu:", value=date(2024, 1, 1))
end_date = st.date_input("Lựa chọn ngày kết thúc:", value=date.today())

if symbol:
    # Fetch the stock data
    df = Quote(symbol=symbol, source='tcbs')
    data = df.history(
        start=start_date.strftime('%Y-%m-%d'), 
        end=end_date.strftime('%Y-%m-%d'), 
        interval='1d'
    )
    
   
    data['time'] = pd.to_datetime(data['time'])

    # Calculate SMA and EMA    
    data['SMA_20'] = data['close'].rolling(window=20).mean()  # 30-day Simple Moving Average
    data['SMA_50'] = data['close'].rolling(window=50).mean() 
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()  # 30-day Exponential Moving Average
    data['BB_upper'] = data['SMA_20'] + 2 * data['close'].rolling(window=20).std()  # Bollinger Band Upper
    data['BB_lower'] = data['SMA_20'] - 2 * data['close'].rolling(window=20).std()  # Bollinger Band Lower

    # Display the dataframe
    st.write(f"Dữ liệu chứng khoán cho mã {symbol} từ ngày {start_date} đến ngày {end_date}:")
    st.dataframe(data[['time','open','high','low', 'close','volume']])


    # Checkboxes for SMA and EMA
    col1, col2 = st.columns(2)  # Create two columns
    with col1:
         show_sma = st.checkbox("Đường Chỉ báo SMA 20", value=True)
         show_sma_50 = st.checkbox("Đường Chỉ báo SMA 50", value=True)
         show_rsi = st.checkbox("Chỉ báo RSI", value=False)
    with col2:
         show_ema = st.checkbox("Đường Chỉ báo EMA 20", value=True)
         show_bb = st.checkbox("Đường Chỉ báo Bollinger Bands", value=True)
         show_macd = st.checkbox("Chỉ báo MACD", value=False)

    # Display the time series graph using Matplotlib
    st.write("Đồ thị chứng khoán theo thời gian:")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data['time'], data['close'], label='Giá đóng cửa', color='blue')
    
    if show_sma:
        ax.plot(data['time'], data['SMA_20'], label='Đường SMA 20', color='orange')
    if show_sma:
        ax.plot(data['time'], data['SMA_50'], label='Đường SMA 50', color='purple')
    if show_ema:
        ax.plot(data['time'], data['EMA_20'], label='Đường EMA 20', color='green')
    if show_bb:
        ax.fill_between(data['time'], data['BB_upper'], data['BB_lower'], color='red', alpha=0.1, label='Bollinger Bands')
        ax.plot(data['time'], data['BB_upper'], label='Đường Bollinger biên trên', color='red', linestyle='--')   
        ax.plot(data['time'], data['BB_lower'], label='Đường Bollinger biên dưới', color='red', linestyle='--')
     
    # Calculate MACD
    if show_macd:
       short_ema = data['close'].ewm(span=12, adjust=False).mean()  # Short-term EMA
       long_ema = data['close'].ewm(span=26, adjust=False).mean()  # Long-term EMA
       data['MACD'] = short_ema - long_ema
       data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    if show_rsi:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

    ax.set_xlabel('Ngày')
    ax.set_ylabel('Giá đóng cửa')
    ax.set_title(f"Giá chứng khoán cho mã {symbol}")
    ax.legend(prop={'size': 6})
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(fig)

    # Plot MACD
    if show_macd:
       st.write("Đồ thị MACD:")
       fig_macd, ax_macd = plt.subplots(figsize=(10, 4))
       ax_macd.plot(data['time'], data['MACD'], label='MACD', color='blue')
       ax_macd.plot(data['time'], data['Signal_Line'], label='Signal Line', color='red', linestyle='--')
       ax_macd.set_xlabel('Ngày')
       ax_macd.set_ylabel('Giá trị MACD')
       ax_macd.set_title('Chỉ số MACD')
       ax_macd.legend()
       plt.xticks(rotation=45)
       st.pyplot(fig_macd)

    # Plot RSI
    if show_rsi:
       st.write("Đồ thị RSI:")
       fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
       ax_rsi.plot(data['time'], data['RSI'], label='RSI', color='blue')
       ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought (70)')
       ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold (30)')
       ax_rsi.set_xlabel('Ngày')
       ax_rsi.set_ylabel('RSI')
       ax_rsi.set_title('Chỉ số RSI')
       ax_rsi.legend()
       plt.xticks(rotation=45)
       st.pyplot(fig_rsi)

    # Add checkboxes for prediction models
    st.markdown(
    "<h2 style='color: darkblue;'>Dự báo giá cổ phiếu trong tương lai</h2>",
    unsafe_allow_html=True
    )
    st.write("Chọn phương pháp dự báo:")

    col1, col2 = st.columns(2)  # Create two columns

    with col1:
        use_lstm = st.checkbox("Dự báo theo LSTM")
        use_rf = st.checkbox("Dự báo theo Random Forest")
    with col2:
        use_arima = st.checkbox("Dự báo theo ARIMA")
        use_prophet = st.checkbox("Dự báo theo Mô hình Prophet")

# Prepare future dates for prediction
    future_days = 30
    future_dates = pd.date_range(start=data['time'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)



# ARIMA Prediction
    if use_arima:    
       arima_model = ARIMA(data['close'], order=(5, 1, 0))
       arima_result = arima_model.fit()
       future_arima_predictions = arima_result.forecast(steps=future_days)
       future_arima_data = pd.DataFrame({'time': future_dates, 'ARIMA': future_arima_predictions})
       ax.plot(future_arima_data['time'], future_arima_data['ARIMA'], label='Dự báo ARIMA', color='red', linestyle='--')
       


# Prophet Prediction
    if use_prophet:       
       prophet_data = data[['time', 'close']].rename(columns={'time': 'ds', 'close': 'y'})
       prophet_model = Prophet()
       prophet_model.fit(prophet_data)
       future_prophet = prophet_model.make_future_dataframe(periods=future_days)
       forecast = prophet_model.predict(future_prophet)
       future_prophet_data = forecast[['ds', 'yhat']].tail(future_days).rename(columns={'ds': 'time', 'yhat': 'Prophet'})
       ax.plot(future_prophet_data['time'], future_prophet_data['Prophet'], label='Dự báo Prophet', color='brown', linestyle='--')


# LSTM Prediction
    if use_lstm:
    # Preprocess data for LSTM
       scaler = MinMaxScaler(feature_range=(0, 1))
       scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    # Prepare training data
       look_back = 60  # Use the last 60 days to predict the next day
       X_train, y_train = [], []
       for i in range(look_back, len(scaled_data)):
           X_train.append(scaled_data[i - look_back:i, 0])
           y_train.append(scaled_data[i, 0])
       X_train, y_train = np.array(X_train), np.array(y_train)
       X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
       lstm_model = Sequential()
       lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
       lstm_model.add(LSTM(units=50, return_sequences=False))
       lstm_model.add(Dense(units=25))
       lstm_model.add(Dense(units=1))

    # Compile and train the model
       lstm_model.compile(optimizer='adam', loss='mean_squared_error')
       lstm_model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=0)

    # Prepare test data for prediction
       test_data = scaled_data[-look_back:]
       X_test = []
       X_test.append(test_data)
       X_test = np.array(X_test)
       X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict future prices
       lstm_predictions = []
       for _ in range(future_days):
           predicted_price = lstm_model.predict(X_test)
           lstm_predictions.append(predicted_price[0, 0])
        # Update test data with the predicted price
           test_data = np.append(test_data[1:], predicted_price)
           X_test = np.array([test_data])
           X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Inverse transform predictions to original scale
       lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
       future_lstm_data = pd.DataFrame({'time': future_dates, 'LSTM': lstm_predictions.flatten()})

    # Plot LSTM predictions
       ax.plot(future_lstm_data['time'], future_lstm_data['LSTM'], label='Dự báo LSTM', color='cyan', linestyle='--')

    # Add LSTM predictions to the combined table
       if 'combined_predictions' in locals():
           combined_predictions['LSTM'] = future_lstm_data['LSTM'].values
       else:
           combined_predictions = pd.DataFrame({'time': future_dates, 'LSTM': future_lstm_data['LSTM'].values})

    if use_rf:
        look_back = 60  # Use the last 60 days to predict the next day
        X_rf, y_rf = [], []
        for i in range(look_back, len(data['close'])):
            X_rf.append(data['close'].iloc[i - look_back:i].values)
            y_rf.append(data['close'].iloc[i])
        X_rf, y_rf = np.array(X_rf), np.array(y_rf)

    # Train the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_rf, y_rf)

      # Predict future prices
        rf_predictions = []
        last_look_back = data['close'].iloc[-look_back:].values.reshape(1, -1)
        for _ in range(future_days):
            predicted_price = rf_model.predict(last_look_back)[0]
            rf_predictions.append(predicted_price)
        # Update the input for the next prediction
            last_look_back = np.append(last_look_back[:, 1:], [[predicted_price]], axis=1)

    # Create a DataFrame for Random Forest predictions
        future_rf_data = pd.DataFrame({'time': future_dates, 'Random Forest': rf_predictions})

    # Plot Random Forest predictions
        ax.plot(future_rf_data['time'], future_rf_data['Random Forest'], label='Dự báo Random Forest', color='purple', linestyle='--')
   # Add Random Forest predictions to the combined table
        if 'combined_predictions' in locals():
            combined_predictions['Random Forest'] = future_rf_data['Random Forest'].values
        else:
            combined_predictions = pd.DataFrame({'time': future_dates, 'Random Forest': future_rf_data['Random Forest'].values})    

# Combine and display predictions from selected methods in one table
if any([use_arima, use_prophet, use_lstm, use_rf]):  # Check if any method is selected
    # Initialize a DataFrame with future dates
    combined_predictions = pd.DataFrame({'time': future_dates})

    # Add predictions from selected methods
    if use_arima:
        combined_predictions['ARIMA'] = future_arima_data['ARIMA'].values
    if use_prophet:
        combined_predictions['Prophet'] = future_prophet_data['Prophet'].values
    if use_lstm:
        combined_predictions['LSTM'] = future_lstm_data['LSTM'].values
    if use_rf:
        combined_predictions['Random Forest'] = future_rf_data['Random Forest'].values

    # Display the combined table
    st.write("Bảng tổng hợp dự báo từ các phương pháp đã chọn:")
    st.table(combined_predictions.head(10))
else:
    st.write("")

# Finalize the plot
ax.legend(prop={'size': 6})
plt.xticks(rotation=45)
st.pyplot(fig)


# Add custom CSS for the button
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #003366; /* Dark blue background */
        color: white; /* White text */
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    div.stButton > button:hover {
        background-color: #002244; /* Slightly darker blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Button to calculate the average prediction
if st.button("Dự báo kết hợp"):
    # Combine predictions into a single DataFrame
    combined_predictions = pd.DataFrame({'time': future_dates[:10]})  # Use only the next 10 days
    if use_arima:
        combined_predictions['ARIMA'] = future_arima_data['ARIMA'][:10].values
    if use_prophet:
        combined_predictions['Prophet'] = future_prophet_data['Prophet'][:10].values
    if use_lstm:
        combined_predictions['LSTM'] = future_lstm_data['LSTM'][:10].values
    if use_rf:
        combined_predictions['Random Forest'] = future_rf_data['Random Forest'][:10].values
    # Calculate the average prediction
    combined_predictions['Average'] = combined_predictions.iloc[:, 1:].mean(axis=1)

    # Display the results in a table
    st.write("Bảng dự báo kết hợp từ các phương pháp đã chọn:")
    st.table(combined_predictions)

# Function to get buy/sell recommendation from Gemini API
def get_gemini_recommendation(symbol, summary):
    """
    Fetch buy/sell recommendation from Gemini AI based on the summary of the last 30 days.
    """
    # Set the Google Generative AI API key (authentication)
    API_KEY = "AIzaSyDgpMnXlUyC-Ebi7z4xRkdmxBzvfWkcw2Q"

    # Authenticate with Google Generative AI
    genai.configure(api_key=API_KEY)

    # Define the generative model
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Create a prompt for the AI model
    prompt = (
        f"Cho mã cổ phiếu '{symbol}' và tổng kết giá cổ phiếu trong 30 ngày gần nhất:\n"
        f"- Giá trung bình: {summary['Average Price']:.2f}\n"
        f"- Giá cao nhất: {summary['Highest Price']:.2f}\n"
        f"- Giá thấp nhất: {summary['Lowest Price']:.2f}\n"
        f"- Giá gần nhất: {summary['Latest Price']:.2f}\n"
        "Đưa ra lời khuyên mua, bán hay giữ cổ phiếu này trong 3 ngày tới, trong trung hạn và trong dài hạn."
        "Giải thích lý do đằng sau những lời khuyên."
    )

    # Generate a response from the AI model
    try:
        response = model.generate_content(prompt)
        recommendation = response.text.strip()
    except Exception as e:
        recommendation = f"Error fetching recommendation: {str(e)}"

    return recommendation


def summarize_timeline_and_recommend(data, symbol):
    """
    Summarize the last 30 days of stock data and get a buy/sell recommendation from Gemini AI.
    """
    # Filter the last 30 rows
    last_30_days = data.tail(30)

    # Calculate summary statistics
    avg_price = last_30_days['close'].mean()
    max_price = last_30_days['close'].max()
    min_price = last_30_days['close'].min()
    latest_price = last_30_days['close'].iloc[-1]

    # Create a summary dictionary
    summary = {
        "Average Price": avg_price,
        "Highest Price": max_price,
        "Lowest Price": min_price,
        "Latest Price": latest_price,
    }

    # Get a recommendation from Gemini AI
    recommendation = get_gemini_recommendation(symbol, summary)
    summary["Recommendation"] = recommendation

    return summary


# Display summary and recommendation
st.markdown(
    "<h2 style='color: darkblue;'>Lời khuyên từ AI về đầu tư cho mã cổ phiếu</h2>",
    unsafe_allow_html=True
)


# Button to fetch recommendation
if st.button("Lời khuyên AI"):
    if symbol and not data.empty:
        # Summarize data and get recommendation
        summary = summarize_timeline_and_recommend(data, symbol)

        # Display summary
        st.write(f"Tổng quan giá cổ phiếu {symbol} trong 30 ngày gần nhất:")
        st.write(f"- Giá trung bình: {summary['Average Price']:.2f}")
        st.write(f"- Giá cao nhất: {summary['Highest Price']:.2f}")
        st.write(f"- Giá thấp nhất: {summary['Lowest Price']:.2f}")
        st.write(f"- Giá gần nhất: {summary['Latest Price']:.2f}")
        st.markdown(
            f"<h6>Lời khuyên từ AI: {summary['Recommendation']}</h6>",
            unsafe_allow_html=True
        )
    else:
        st.write("Vui lòng nhập mã cổ phiếu và đảm bảo dữ liệu không trống.")
