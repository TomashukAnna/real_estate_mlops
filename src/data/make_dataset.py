#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """Загрузка и первичная очистка данных"""
    df = pd.read_csv(filepath)
    
    #преобразовываем данные к нужным типам
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday_number'] = df['date'].dt.weekday
    df.drop('date', axis=1, inplace=True)
    
    df.drop('time', axis=1, inplace=True)
    
    df['region'] = df['region'].astype('object')
    df['building_type'] = df['building_type'].astype('object')
    df['object_type'] = df['object_type'].astype('object')

    
    # Фильтрация по региону (СПб)
    df = df[(df['region'] == 3446) | (df['region'] == 2661)] #отбираем санкт-петербург и лен. область
    
    #перезапишем индексы
    df = df.reset_index(drop=True)
    
    
    # Удаление дубликатов
    columns_for_deduplication = [
        'price', 'geo_lat', 'geo_lon', 'region',
        'building_type', 'level', 'levels', 'rooms',
        'area', 'kitchen_area', 'object_type']
    existing_columns = [col for col in columns_for_deduplication if col in df.columns]
    df = df.drop_duplicates(subset=existing_columns, keep='first')    
    
    
    # Добавляем цену за кв.м
    df['price_per_m2'] = df['price'] / df['area']
        
    return df

if __name__ == '__main__':
    df = load_and_clean_data('data/raw/russia_real_estate.csv')
    df.to_csv('data/processed/cleaned_data.csv', index=False)

