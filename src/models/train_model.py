#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://localhost:5000')

# Загружаем данные
df = pd.read_csv('data/processed/cleaned_data.csv')

# Признаки
features = ['region', 'building_type', 'level', 'levels', 'year', 'month', 'rooms', 'area', 'kitchen_area', 'object_type', 'weekday_number']
target = 'price_per_m2'

# Убираем пропуски
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Разделяем по годам для демонстрации дрейфа
X_train = df[df['year'] <= 2024][features]
y_train = df[df['year'] <= 2024][target]
X_test = df[df['year'] == 2025][features]
y_test = df[df['year'] == 2025][target]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Обучение с MLflow
with mlflow.start_run(run_name="random_forest_v1"):
    # Параметры
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("features", features)
    mlflow.log_param("train_years", "<=2024")
    mlflow.log_param("test_years", "2025")
    
    # Обучение
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Метрики
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Сохраняем модель
    mlflow.sklearn.log_model(model, "model")
    
    print(f"MAE: {mae:,.0f}")
    print(f"MAPE: {mape:.3f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"R2: {r2:.3f}")

