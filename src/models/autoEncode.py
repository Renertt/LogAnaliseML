import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def train_autoencoder(features):

    print("Filtering data")
    data_numeric = features.select_dtypes(include=[np.number]).fillna(0)

    print("Scaling data")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    joblib.dump(scaler, 'modelsSaved/scaler.save') 

    input_dim = data_scaled.shape[1] # 7 признаков
    
    # Бутылочное горлышко в 30% данных
    encoding_dim = int(input_dim * 0.3)
    if encoding_dim < 1: encoding_dim = 1 # Защита, если признаков совсем мало

    input_layer = Input(shape=(input_dim,))
    
    # Сжимаем
    hidden_1 = Dense(64, activation="relu")(input_layer)
    hidden_2 = Dense(32, activation="relu")(hidden_1)
    bottleneck = Dense(encoding_dim, activation="relu")(hidden_2)
    
    # Разжимаем
    decoder_hidden_1 = Dense(32, activation="relu")(bottleneck)
    decoder_hidden_2 = Dense(64, activation="relu")(decoder_hidden_1)
    output_layer = Dense(input_dim, activation="linear")(decoder_hidden_2)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    print("Starting training")
    autoencoder.fit(
        data_scaled, data_scaled,
        epochs=30,           
        batch_size=16,       
        shuffle=True,        
        validation_split=0.1,
        verbose=1
    )

    print("Saving model")
    autoencoder.save('modelsSaved/autoencoder.keras')
    print("Done")
    return autoencoder

def detect_anomalies(features_df, model, threshold, plot_mse):

    scaler = joblib.load('modelsSaved/scaler.save')

    ips = features_df['ip'].values
    
    data_numeric = features_df.select_dtypes(include=[np.number]).fillna(0)
    data_scaled = scaler.transform(data_numeric)

    reconstructions = model.predict(data_scaled)

    # Считаем ошибку восстановления (MSE) для каждой строки
    # (Оригинал - Восстановленное)^2 -> среднее
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)

    if threshold is None:
        threshold = np.percentile(mse, 85)
        print(f"Auto threshold: {threshold}")

    anomalies_mask = mse > threshold

    if plot_mse:
        plt.figure(figsize=(12, 6))
        plt.hist(mse, bins=100, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', label=f'threshold={threshold}')
        plt.title('MSE Distribution (Anomaly Detection)')
        plt.xlabel('MSE')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.savefig('data/processed/mse_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return anomalies_mask, mse

def AE_load_model(model_path):
    return tf.keras.models.load_model(model_path)