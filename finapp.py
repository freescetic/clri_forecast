import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA

# --- Styling ---
st.set_page_config(
    page_title="Hide Scope CLRI Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)

# Reading the CSV file into a DataFrame
df = pd.read_excel('CLRI FMCG ML Model Dataset Ajay Vikhram S M and Diwin Joshua.xlsx')  # Replace with your actual file path

# Function to forecast using LSTM
def forecast_lstm(df_country, start_year, end_year):
    # Preparing data for LSTM
    data = df_country['Export'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Creating training dataset
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 3  # Number of previous years to consider
    X_train, Y_train = create_dataset(train_data, look_back)
    X_test, Y_test = create_dataset(test_data, look_back)

    # Reshaping input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=500, batch_size=1, verbose=0)

    # Make predictions on the training data
    train_predict = model.predict(X_train)
    train_predict = scaler.inverse_transform(train_predict)

    # Forecasting future values
    future_predictions = []
    last_values = data[-look_back:]  # Last 'look_back' values
    forecast_years = end_year - start_year + 1
    for _ in range(forecast_years):  # Forecast for specified years
        future_predict = model.predict(np.reshape(last_values, (1, look_back, 1)))
        future_predictions.append(future_predict[0, 0])
        last_values = np.append(last_values[1:], future_predict[0, 0])  # Update last_values

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten(), train_predict  # Return both predictions

# Function to forecast using ARIMA
def forecast_arima(df_country, start_year, end_year, p=5, d=1, q=0):
    model = ARIMA(df_country['Export'], order=(p, d, q))
    model_fit = model.fit()

    # Make predictions on the training data
    train_predict = model_fit.predict(start=p, end=len(df_country)-1)  # Adjust start for ARIMA

    forecast_years = end_year - start_year + 1
    forecast = model_fit.predict(start=len(df_country), end=len(df_country) + forecast_years - 1)
    return forecast.values.flatten(), train_predict  # Return both predictions

# Streamlit app
def main():
    st.title("Export Forecast Dashboard")

    tab1, tab2 = st.tabs(["Predict", "Simulate"])

    with tab1:
        st.header("Predict Export Values")

        # Model selection
        model_choice = st.radio("Select Forecasting Model:", ["ARIMA", "LSTM"])

        # Country selection
        countries = ['UAE', 'Netherlands', 'Germany', 'Zimbabwe', 'South Africa', 'Botswana', 'Uganda']
        country_name = st.selectbox("Select Country:", countries)

        # Year selection
        start_year = st.number_input("Start Year:", min_value=2023, max_value=2040, value=2023)
        end_year = st.number_input("End Year:", min_value=start_year, max_value=2040, value=2032)

        # Figsize input
        figsize_width = st.number_input("Figure Width:", min_value=5, max_value=20, value=10)
        figsize_height = st.number_input("Figure Height:", min_value=5, max_value=20, value=6)

        if st.button("Forecast"):
            df_country = df[df['Year'] <= 2022].copy()

            df_country = df_country[[
                'Year',
                f'Export to {country_name} from India in US$ Thousand (Hides and Skins)',
                f'GDP Nominal in USD of {country_name}',
                f'GDP Per Capita in {country_name} in USD',
                f'Inflation Rate in {country_name}',
                f'Fuel Import in {country_name} in US$ Thousand',
                f'Food Import in {country_name} in US$ Thousand',
            ]]

            df_country.columns = ['Year', 'Export', 'GDP', 'GDP_Per_Capita', 'Inflation_Rate', 'Fuel_Import', 'Food_Import']

            # Interpolate missing values in all relevant columns
            for col in ['Export', 'GDP', 'GDP_Per_Capita', 'Inflation_Rate', 'Fuel_Import', 'Food_Import']:
                df_country[col] = df_country[col].interpolate(method='linear')

            if model_choice == "ARIMA":
                forecast, train_predict = forecast_arima(df_country, start_year, end_year)
                # Plot the forecast (ARIMA)
                plt.figure(figsize=(figsize_width, figsize_height))
                plt.plot(df_country['Year'], df_country['Export'], label='Actual')
                plt.plot(df_country['Year'][5:], train_predict, label='Train Prediction')  # Plot train predictions, starting from year 5 to align with train_predict
                plt.plot(range(start_year, end_year + 1), forecast, label='Forecast')
                plt.xlabel('Year')
                plt.ylabel(f'Export to {country_name} from India in US$ Thousand (Hides and Skins)')
                plt.title(f'Forecast of Export to {country_name} from India in US$ Thousand (Hides and Skins)')
                plt.legend()
                st.pyplot(plt)
            else:  # LSTM
                forecast, train_predict = forecast_lstm(df_country, start_year, end_year)
                # Plot the forecast (LSTM)
                plt.figure(figsize=(figsize_width, figsize_height))
                plt.plot(df_country['Year'], df_country['Export'], label='Actual')

                # Adjust the x-axis values to match the length of train_predict for LSTM
                train_years = df_country['Year'][3:-3]  # Exclude the first 3 and last 3 years
                plt.plot(train_years[:len(train_predict)], train_predict, label='Train Prediction')  # Plot train predictions

                plt.plot(range(start_year, end_year + 1), forecast, label='Forecast')
                plt.xlabel('Year')
                plt.ylabel(f'Export to {country_name} from India in US$ Thousand (Hides and Skins)')
                plt.title(f'Forecast of Export to {country_name} from India in US$ Thousand (Hides and Skins)')
                max_value = max(df_country['Export'].max(), max(forecast))
                plt.ylim(0, max_value * 1.5)
                plt.legend()
                st.pyplot(plt)

            # Display results
            st.subheader("Forecast Results")
            forecast_df = pd.DataFrame({'Year': range(start_year, end_year + 1), 'Forecast Export': forecast})
            st.table(forecast_df)


    with tab2:
        st.header("Simulate Export Scenarios")

        # Country selection
        countries = ['UAE', 'Netherlands', 'Germany', 'Zimbabwe', 'South Africa', 'Botswana', 'Uganda']
        country_name = st.selectbox("Select Country for Simulation:", countries)

        # Fetch country data
        df_country = df[df['Year'] <= 2022].copy()
        df_country = df_country[[
            'Year',
            f'Export to {country_name} from India in US$ Thousand (Hides and Skins)',
            f'GDP Nominal in USD of {country_name}',
            f'GDP Per Capita in {country_name} in USD',
            f'Inflation Rate in {country_name}',
            f'Fuel Import in {country_name} in US$ Thousand',
            f'Food Import in {country_name} in US$ Thousand',
        ]]
        df_country.columns = ['Year', 'Export', 'GDP', 'GDP_Per_Capita', 'Inflation_Rate', 'Fuel_Import', 'Food_Import']

        # Interpolate missing values in all relevant columns
        for col in ['Export', 'GDP', 'GDP_Per_Capita', 'Inflation_Rate', 'Fuel_Import', 'Food_Import']:
            df_country[col] = df_country[col].interpolate(method='linear')

        # Display editable dataframe
        st.subheader("Edit Country Data:")
        edited_df = st.data_editor(df_country)

        # Model selection
        model_choice = st.radio("Select Forecasting Model for Simulation:", ["ARIMA", "LSTM"])

        # Year selection
        start_year = st.number_input("Simulation Start Year:", min_value=2023, max_value=2050, value=2023)
        end_year = st.number_input("Simulation End Year:", min_value=start_year, max_value=2050, value=2032)

        # Figsize input
        figsize_width = st.number_input("Figure Width:", min_value=5, max_value=20, value=10, key="sim_width")
        figsize_height = st.number_input("Figure Height:", min_value=5, max_value=20, value=6, key="sim_height")

        if st.button("Simulate"):
            if model_choice == "ARIMA":
                forecast, train_predict = forecast_arima(edited_df, start_year, end_year)
                # Plot the forecast (ARIMA)
                plt.figure(figsize=(figsize_width, figsize_height))
                plt.plot(edited_df['Year'], edited_df['Export'], label='Actual (Edited)')
                plt.plot(edited_df['Year'][5:], train_predict, label='Train Prediction')  # Plot train predictions, starting from year 5 to align with train_predict
                plt.plot(range(start_year, end_year + 1), forecast, label='Forecast')
                plt.xlabel('Year')
                plt.ylabel(f'Export to {country_name} from India in US$ Thousand (Hides and Skins)')
                plt.title(f'Forecast of Export to {country_name} from India in US$ Thousand (Hides and Skins)')
                plt.legend()
                st.pyplot(plt)
            else:  # LSTM
                forecast, train_predict = forecast_lstm(edited_df, start_year, end_year)
                # Plot the forecast (LSTM)
                plt.figure(figsize=(figsize_width, figsize_height))
                plt.plot(edited_df['Year'], edited_df['Export'], label='Actual (Edited)')

                # Adjust the x-axis values to match the length of train_predict for LSTM
                train_years = edited_df['Year'][3:-3]  # Exclude the first 3 and last 3 years
                plt.plot(train_years[:len(train_predict)], train_predict, label='Train Prediction')  # Plot train predictions

                plt.plot(range(start_year, end_year + 1), forecast, label='Forecast')
                plt.xlabel('Year')
                plt.ylabel(f'Export to {country_name} from India in US$ Thousand (Hides and Skins)')
                plt.title(f'Forecast of Export to {country_name} from India in US$ Thousand (Hides and Skins)')
                max_value = max(edited_df['Export'].max(), max(forecast))
                plt.ylim(0, max_value * 1.5)
                plt.legend()
                st.pyplot(plt)

            # Display results
            st.subheader("Simulation Results")
            forecast_df = pd.DataFrame({'Year': range(start_year, end_year + 1), 'Forecast Export': forecast})
            st.table(forecast_df)

if __name__ == "__main__":
    main()
