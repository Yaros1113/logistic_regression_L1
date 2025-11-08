import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('data/data.csv')

# Общая информация о типах данных
print("\nИнформация о типах данных:")
print(df.info())


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
    
    # Типы данных
    print("\nТИПЫ ДАННЫХ:")
    print(df.dtypes.value_counts())
    
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
        
        # Удаление дубликатов ???
        # df = df.drop_duplicates()
    
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

    # Выбросы
    print("\nВЫБРОСЫ ПО ПРИЗНАКАМ (метод IQR):")
    for col in numeric_columns:
        outliers = find_outliers_iqr(df[col])
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} выбросов ({len(outliers)/len(df)*100:.2f}%)")

    # Визуализация выбросов
    plt.figure(figsize=(12, 6))
    df[numeric_columns].boxplot()
    plt.title('Визуализация выбросов')
    plt.xticks(rotation=45)
    plt.show()
    
    return df

# Поиск выбросов с помощью IQR
def find_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers



input("\nВведите что-нибудь")
exploratory_data_analysis(df, 'target')