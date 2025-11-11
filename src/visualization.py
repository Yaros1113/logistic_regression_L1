import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_data_analysis(df, target_column=None):
    """
    Полный первичный анализ данных
    """
    print("=" * 50)
    print("ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ")
    print("=" * 50)
    
    # Базовая информация
    print(f"Размер данных: {df.shape}")
    print(f"Объекты: {df.shape[0]}, Признаки: {df.shape[1]}")
    
    # Пропущенные значения
    print("\nПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
    print("\nОбщее количество пропущенных значений:", df.isnull().sum().sum(), "\n")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Пропуски': missing,
        'Процент': missing_percent
    })
    print(missing_df[missing_df['Пропуски'] > 0])
    # Визуализация пропущенных признаков
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Карта пропущенных значений')
    plt.show()
    
    # Количество полных дубликатов
    duplicates_count = df.duplicated().sum()
    print(f"\nПОЛНЫЕ ДУБЛИКАТЫ: {duplicates_count}")

    if duplicates_count > 0:
        print("Примеры дубликатов:")
        print(df[df.duplicated()].head())
    
    # Баланс классов (если указана целевая переменная)
    if target_column and target_column in df.columns:
        print(f"\nБАЛАНС КЛАССОВ ({target_column}):")
        print(df[target_column].value_counts())
        print("\nПРОЦЕНТНОЕ РАСПРЕДЕЛЕНИЕ:")
        print(df[target_column].value_counts(normalize=True).round(4) * 100)
        # Визуализация баланса классов
        plt.figure(figsize=(8, 6))
        df[target_column].value_counts().plot(kind='bar')
        plt.title('Распределение классов')
        plt.show()
    
    # Статистика по числовым признакам
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        print(f"\nОПИСАТЕЛЬНАЯ СТАТИСТИКА ({len(numeric_columns)} числовых признаков):")
        print(df[numeric_columns].describe())

    return df

# Загрузка сырых данных
df = pd.read_csv('data/data.csv')

exploratory_data_analysis(df, 'target')
input("\nВведите что-нибудь, что бы увидеть анализ обработанных данных")

# Загрузка обработанных данных
df = pd.read_csv('data/update.csv')
exploratory_data_analysis(df, 'target')