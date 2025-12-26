import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
import joblib

def train_autoencoder(features):
    
    print("Фильтрация данных")
    data_numeric = features.select_dtypes(include=[np.number]).fillna(0)
    
    print("Масштабируем данные")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    joblib.dump(scaler, 'data/temp/scaler.save') 

    input_dim = data_scaled.shape[1] # 7 признаков
    
    # Бутылочное горлышко в 70% данных
    encoding_dim = int(input_dim * 0.7)
    if encoding_dim < 1: encoding_dim = 1 # Защита, если признаков совсем мало

    input_layer = Input(shape=(input_dim,))
    
    # Сжимаем
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    
    # Разжимаем
    decoder = Dense(input_dim, activation="linear")(encoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    print("Начинаем обучение")
    autoencoder.fit(
        data_scaled, data_scaled,
        epochs=30,           
        batch_size=16,       
        shuffle=True,        
        validation_split=0.1,
        verbose=1
    )

    print("Сохраняем модель")
    autoencoder.save('modelsSaved/autoencoder.keras')
    print("Готово")

def detect_anomalies(features_df, threshold=None):
    
    model = tf.keras.models.load_model('modelsSaved/autoencoder.keras')
    scaler = joblib.load('data/temp/scaler.save')

    ips = features_df['ip'].values # Отдельное сохранение айпи
    
    data_numeric = features_df.select_dtypes(include=[np.number]).fillna(0)
    data_scaled = scaler.transform(data_numeric)

    reconstructions = model.predict(data_scaled) # Предсказываем

    # Считаем ошибку восстановления (MSE) для каждой строки
    # (Оригинал - Восстановленное)^2 -> среднее
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)

    # Определяем, кто аномалия
    if threshold is None:
        threshold = np.percentile(mse, 85)
        print(f"Автоматический порог ошибки: {threshold}")

    anomalies_mask = mse > threshold
    
    # Возвращаем DataFrame с аномальными IP и их ошибкой
    results = pd.DataFrame({
        'ip': ips,
        'anomaly_score': mse,
        'is_anomaly': anomalies_mask
    })
    
    # Отфильтруем, вернем только плохих парней
    anomalies = results[results['is_anomaly'] == True].sort_values(by='anomaly_score', ascending=False)
    
    return anomalies