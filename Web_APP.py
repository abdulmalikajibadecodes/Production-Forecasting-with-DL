import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from openpyxl import Workbook
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
# import pickle
import time
import io

st.set_page_config(page_title="Oil Production Forecasting", page_icon=":chart_with_upwards_trend:")

# Define function to collect data from user
def get_user_data():
    uploaded_file = st.file_uploader("Upload a CSV file containing production data history", type=["csv", "xlsx","txt"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("First 10 rows of the uploaded data:")
        st.write(data)
        target_col = st.selectbox("Select the target column", data.columns)
        date_col = st.selectbox("Select the date column", data.columns)
        data[date_col] = pd.to_datetime(data[date_col])
        data.set_index(date_col, inplace=True)
        data.sort_index(inplace=True)
        avg_well_head_pressure = st.selectbox("Select the avg_well_head_pressure column", data.columns)
        avg_choke_size = st.selectbox("Select the avg_choke_size column", data.columns)
        avg_annulus_pressure =st.selectbox("Select the avg_annulus_pressure column", data.columns)
        avg_downhole_pressure = st.selectbox("Select the avg_downhole_pressure column", data.columns)
        on_stream_hours = st.selectbox("Select the on_stream_hours column", data.columns)
        feature_col = [avg_well_head_pressure, avg_annulus_pressure, avg_choke_size, avg_downhole_pressure, on_stream_hours]
        X = data[feature_col]
        y = data[target_col]
        # Scale the data using a MinMaxScaler
        input_scaler = StandardScaler()
        output_scaler = StandardScaler()
        X_scaled = input_scaler.fit_transform(X)
        y_scaled = output_scaler.fit_transform(data[[target_col]])
        n_train = int(len(data) * 0.9)
        train_X_scaled, test_X_scaled = X_scaled[:n_train, :], X_scaled[n_train:, :]
        train_y_scaled, test_y_scaled = y_scaled[:n_train, :], y_scaled[n_train:, :]
 
        # Define the training and test data as numpy arrays
        X_train = []
        y_train = []
        lookback = 20
        for i in range(lookback, len(train_X_scaled)):
            X_train.append(train_X_scaled[i-lookback:i, :])
            y_train.append(train_y_scaled[i:i+1, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_test = []
        y_test = []
        for i in range(lookback, len(test_X_scaled)):
            X_test.append(test_X_scaled[i-lookback:i, :])
            y_test.append(test_y_scaled[i:i+1, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Determine the number of features
        n_features = X_train.shape[2]

        return X.iloc[n_train:,:], X_train, y_train, X_test, y_test, input_scaler, output_scaler, n_features,data
    else:
        return None


def create_lstm_model(X_train, X_test, y_train, y_test, input_scaler, output_scaler, data):
    # Define the LSTM model
    np.random.seed(42)
    tf.random.set_seed(42)
    model = Sequential()
    model.add(LSTM(38, input_shape=(X_train.shape[1],X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', run_eagerly = True)

    # Train the model and show a progress bar
    with st.spinner('Training the model...'):
        model.fit(X_train,y_train)
        st.success('Training complete!')

    # Calculate the MAE and RMSE scores
    y_pred_test = model.predict(X_test)
    y_pred_test = output_scaler.inverse_transform(y_pred_test)
    y_pred_train = model.predict(X_train)
    y_pred_train = output_scaler.inverse_transform(y_pred_train)
    mae_score = mae(y_test, y_pred_test)
    rmse_score = mse(y_test, y_pred_test, squared = False)
    y_train = output_scaler.inverse_transform(y_train)
    y_test = output_scaler.inverse_transform(y_test)


    # Show the MAE and RMSE scores
    st.write("MAE score:", round(mae_score,2))
    st.write("RMSE score:", round(rmse_score, 2))

    # fig, axs = plt.subplots(nrows=2, ncols=1,figsize=(14,9))
    # axs[0].scatter(data.index[-len(y_train):], y_train, label="Actual Target", color='orange')
    # axs[0].plot(data.index[-len(y_train):], y_pred_train, label="Predicted Target", color='blue')
    # axs[1].scatter(data.index[-len(y_test):], y_test, label="Actual Test", color="green")
    # axs[1].plot(data.index[-len(y_test):], y_pred_test, label="Predicted Test", color="black")
    # axs[0].legend()
    # axs[1].legend()
    # axs[0].set_xlabel("Date")
    # axs[0].set_ylabel("Oil Production Volume")
    # axs[1].set_xlabel("Date")
    # axs[1].set_ylabel("Oil Production Volume")
    # st.pyplot(fig)

    # filename='lstm_model.joblib'
    # joblib.dump(model,filename)
        # Load the model architecture from a file
    with st.spinner('Saving model...'):
        model.save('user_model')
        st.success('Model saved!')
    
    return model

def forecast_reshape(X,user_input, lookback, input_scaler, num_features):
    
    X_new = X.copy()
    new_index=X_new.index[-1]+pd.DateOffset(days=1)
    
    new_row_df = pd.DataFrame([user_input], columns=X_new.columns, index=[new_index])
    X_new=pd.concat([X_new,new_row_df])
    
    X_forecast = X_new.iloc[-lookback:]
    X_forecast_scaled = input_scaler.fit_transform(X_forecast)
    
    X_forecast_mod = X_forecast_scaled.reshape(1,lookback,num_features)
    
    return X_forecast_mod
    





# Define main function to run the Streamlit app
def main():
    st.sidebar.title("Navigation")
    # app_mode = st.sidebar.selectbox("Select a page", ["Homepage", "Oil Production Forecasting", "Oil Production Prediction"])
    app_mode = st.sidebar.selectbox("Select a page", ["Homepage", "Oil Production Forecasting"])

    if app_mode == "Homepage":
        st.title("WELCOME TO THE ML OIL PRODUCTION FORECASTING WEB APP")
        st.write("### By: ABDULMALIK AYOMIDE AJIBADE")
        st.image("https://i.pinimg.com/564x/fc/f4/45/fcf445c1d77320e13689a01bd8d93eba.jpg", width=700)
        # st.title("Welcome to the Oil Production Forecasting App")
        st.write("Please select a page on the left sidebar to continue.")

    elif app_mode == "Oil Production Forecasting":
        # Call the get_user_data() function to collect data from user
        # Call the oil_forecasting() function to create the page layout
        st.header("Oil Production Forecasting")
        user_data = get_user_data()
        if user_data is not None:
            X_last, X_train, y_train, X_test, y_test, input_scaler, output_scaler, n_features,data = user_data
             # Train model button
            if st.button("Train model", key="train"):
            # Call the create_lstm_model() function to create and train the LSTM model
                model= create_lstm_model(X_train, X_test, y_train, y_test, input_scaler, output_scaler, data)
                joblib.dump(model, 'lstm_model.pkl')
   
    # Create a form to input feature values
            st.write("## Predict Oil Volume")
            Avg_well_head_pressure = st.number_input("Avg_well_head_pressure")
            Avg_annulus_pressure = st.number_input("Avg_annulus_pressure")
            Avg_choke_size= st.number_input("Avg_choke_size")
            Avg_downhole_pressure = st.number_input("Avg_downhole_pressure")
            On_stream_hours = st.number_input("On_stream_hours")

                        
            if st.button('Predict Oil volume'):
                      
                    #X_train, y_train, X_test, y_test, scaler, n_features = user_data
                    input_features = [Avg_well_head_pressure, Avg_annulus_pressure, Avg_choke_size, Avg_downhole_pressure, On_stream_hours]

                    X_pred = forecast_reshape(X_last,input_features, 20, input_scaler, n_features)

                    lstm_model = joblib.load('lstm_model.pkl')
                    
                    scaled_prediction = lstm_model.predict(X_pred)

                    prediction = output_scaler.inverse_transform(scaled_prediction)
                    prediction = np.round(prediction[0], 2)

                    #Display predicted value
                    st.write(f"Forecasted Oil Volume: {round(float(prediction),2)} bbl" )


                    # Convert the prediction to an Excel file
                    df = pd.DataFrame(prediction, columns=["Predicted Oil Volume"])
                    file_name = "predicted_oil_volume.xlsx"

                    # Save the Excel file to a buffer
                    buffer = io.BytesIO()
                    df.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)

                    # Create a download button
                    st.download_button(
                        label="Download Prediction",
                        data=buffer,
                        file_name=file_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        
                # # Add a button to save the model
                # if st.button('Save Model'):
                #     with st.spinner('Saving the model...'):
                #         with open('lstm_model_1.pickle', 'wb') as f:
                #             pickle.dump(model, f)
                #     st.success('Model saved successfully')



    # elif app_mode == "Oil Production Prediction":
    #     # Call the oil_production_prediction() function to create the page layout
    #     st.header("Oil Production Prediction")
    #     filename = 'lstm_model.joblib'
    #     with open('lstm_model_1.pickle', 'rb') as f:
    #         model = pickle.load(f)
    #     predict(model)


# Call the main() function to run the Streamlit app
if __name__ == "__main__":
    main()