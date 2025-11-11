# Импорт библиотек - scikit-learn это полное название, sklearn - стандартное сокращение
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def data_preprocessing():
    """
    Анализ и предварительная обработка входной выборки
    """
    # =============================================================================
    # 2.1 ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
    # =============================================================================

    # Загружаем данные из CSV-файла
    df = pd.read_csv("data/data.csv")

    print("=== ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ ===")
    print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Типы данных:\n{df.dtypes}")

    # Показываем первые 5 строк для ознакомления с данными
    print("\nПервые 5 строк данных:")
    print(df.head())

    # Анализ целевой переменной (что мы хотим предсказать)
    print("\n=== АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ===")
    class_distribution = df['target'].value_counts()
    print("Распределение классов:")
    print(class_distribution)
    print(f"Баланс классов: {class_distribution[0]/(class_distribution[0]+class_distribution[1]):.2%} vs {class_distribution[1]/(class_distribution[0]+class_distribution[1]):.2%}")

    # Удаляем нечисловые признаки (модель работает только с числами)
    print("\n=== ОБРАБОТКА ПРИЗНАКОВ ===")
    non_numeric_cols = []
    for column_name in df.columns:
        # Проверяем, является ли признак числовым и не является ли целевой переменной
        if column_name != 'target' and df[column_name].dtype.name not in ['float64', 'int64']:
            non_numeric_cols.append(column_name)
            df = df.drop(column_name, axis=1)  # axis=1 означает удаление столбца

    print(f"Удалены нечисловые признаки: {non_numeric_cols}")

    # Анализ пропущенных значений
    print("\n=== АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
    missing_percentage = df.isnull().mean() * 100
    print("Процент пропущенных значений по столбцам:")
    for col, percentage in missing_percentage.items():
        if percentage > 0:
            print(f"  {col}: {percentage:.2f}%")

    # Проверка на дубликаты (полностью одинаковые строки)
    print("\n=== ПРОВЕРКА ДУБЛИКАТОВ ===")
    duplicates_count = df.duplicated().sum()
    print(f"Найдено полных дубликатов: {duplicates_count}")

    # =============================================================================
    # 2.2 ПРЕДОБРАБОТКА ДАННЫХ
    # =============================================================================

    print("\n=== ПРЕДОБРАБОТКА ДАННЫХ ===")

    # Удаление дубликатов
    if duplicates_count > 0:
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        print(f"Удалено дубликатов: {removed_duplicates}")
    else:
        print("Дубликаты не найдены, удаление не требуется")

    # Заполнение пропущенных значений
    print("\nЗаполнение пропущенных значений...")
    for column_name in df.columns:
        missing_count = df[column_name].isnull().sum()
        if missing_count > 0:
            # Заполняем пропуски медианой (устойчива к выбросам)
            median_value = df[column_name].median()
            df[column_name] = df[column_name].fillna(median_value)
            print(f"  Заполнено {missing_count} пропусков в '{column_name}' медианой: {median_value:.2f}")

            # Заполняем пропуски 0
            #df[column_name] = df[column_name].fillna(0)
            #print(f"  Заполнено {missing_count} пропусков в '{column_name}' нулями")

    # Сохранение обработанной выборки в файл
    df.to_csv('data/update.csv', index=False)


# Работа с сырыми данными
#data_preprocessing()

# =============================================================================
# 2.3 РАЗБИЕНИЕ НА ВЫБОРКИ И СТАНДАРТИЗАЦИЯ
# =============================================================================

print("\n=== РАЗБИЕНИЕ НА ВЫБОРКИ ===")

# Загрузка предобработанной выборки
df = pd.read_csv("data/update.csv")

# Разделяем данные на признаки (X) и целевую переменную (y)
# X - это все столбцы кроме 'target' (признаки для прогноза)
# y - это столбец 'target' (то, что мы хотим предсказать)
X = df.drop('target', axis=1)
y = df['target']

print(f"Признаки (X): {X.shape[1]} столбцов")
print(f"Целевая переменная (y): {len(y)} значений")

# Разбиваем данные на обучающую и тестовую выборки
# test_size=0.2 означает 20% данных пойдет на тестирование, 80% на обучение
# random_state=42 обеспечивает воспроизводимость результатов
# stratify=y сохраняет распределение классов в обеих выборках
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} объектов")
print(f"Тестовая выборка: {X_test.shape[0]} объектов")
print(f"Распределение классов в обучающей выборке: {y_train.value_counts().to_dict()}")
print(f"Распределение классов в тестовой выборке: {y_test.value_counts().to_dict()}")

print("\n=== СТАНДАРТИЗАЦИЯ ДАННЫХ ===")

# Создаем объект для стандартизации
scaler = StandardScaler()

# Обучаем стандартизатор на обучающих данных и преобразуем их
X_train_scaled = scaler.fit_transform(X_train)

# Преобразуем тестовые данные используя параметры с обучающих данных
X_test_scaled = scaler.transform(X_test)

print("Стандартизация завершена:")
print("  - Все признаки теперь имеют среднее = 0 и стандартное отклонение = 1")
print("  - Это помогает модели работать лучше, т.к. все признаки в одном масштабе")

# =============================================================================
# 2.4 РЕАЛИЗАЦИЯ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ
# =============================================================================

print("\n=== РЕАЛИЗАЦИЯ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ ===")

class LogisticRegression:
    """
    Реализация логистической регрессии с нуля
    Логистическая регрессия - это алгоритм для бинарной классификации (2 класса)
    """
    
    def __init__(self, learning_rate=0.01, n_iters=1000, class_weight=None):
        """
        Инициализация модели
        
        Параметры:
        learning_rate - скорость обучения (какой шаг делать при обновлении весов)
        n_iters - количество итераций обучения
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None  # веса для каждого признака
        self.bias = None     # смещение (базовое значение)
        self.class_weight = class_weight
        self.loss_history = []
        self.accuracy_history = []
    
    def compute_loss(self, y_true, y_pred, sample_weights=None):
        """
        Вычисление функции потерь (Функционал качества) (Binary Cross-Entropy)
        """
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)  # Зажимает между мин, макс
        
        if sample_weights is not None:
            loss = -np.mean(
                sample_weights * (y_true * np.log(y_pred_clipped) + 
                                 (1 - y_true) * np.log(1 - y_pred_clipped))
            )
        else:
            loss = -np.mean(
                y_true * np.log(y_pred_clipped) + 
                (1 - y_true) * np.log(1 - y_pred_clipped)
            )
        
        return loss
    
    def sigmoid(self, z):
        """
        Сигмоидная функция - преобразует любое число в диапазон от 0 до 1
        
        Формула: sigmoid(z) = 1 / (1 + e^(-z))
        
        Используется для получения вероятности принадлежности к классу 1
        """
        # Защита от переполнения (очень больших чисел)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_class_weights(self, y):
        """Вычисление весов классов для борьбы с дисбалансом"""
        if self.class_weight == 'balanced':
            # Формула из sklearn: n_samples / (n_classes * np.bincount(y))
            n_samples = len(y)
            n_classes = len(np.unique(y))
            class_counts = np.bincount(y)
            weights = n_samples / (n_classes * class_counts)
            return {0: weights[0], 1: weights[1]}
        else:
            return self.class_weight or {0: 1, 1: 1}  # По умолчанию равные веса
        
    def compute_gradients(self, X, y_true, y_pred, sample_weights=None):
        """
        Вычисление градиентов из функции потерь

        Функция потерь неявно используется в обучении через градиенты.
        """
        error = y_pred - y_true  # Градиент функции потерь
        
        if sample_weights is not None:
            weighted_error = error * sample_weights
        else:
            weighted_error = error
        
        n_samples = X.shape[0]
        # Градиент по весам: dJ/dw = (1/m) * Xᵀ * (ŷ - y)
        # X.T - транспонированная матрица признаков
        # np.dot(X.T, error) - сумма произведений ошибок на значения признаков
        dw = (1 / n_samples) * np.dot(X.T, weighted_error)

        # Градиент по смещению: dJ/db = (1/m) * Σ (ŷ - y)
        # Суммируем все ошибки
        db = (1 / n_samples) * np.sum(weighted_error)
        
        return dw, db
    
    def fit(self, X, y):
        """
        Обучение модели на данных
        
        Параметры:
        X - матрица признаков (строки - объекты, столбцы - признаки)
        y - вектор целевых значений (0 или 1)
        """
        n_samples, n_features = X.shape
        
        # Инициализация весов нулями
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        self.accuracy_history = []

        # Вычисляем веса классов
        class_weights = self.compute_class_weights(y)
        print(f"\nВеса классов: {class_weights}")
        
        # Создаем вектор весов для каждого образца
        sample_weights = np.array([class_weights[yi] for yi in y])

        print(f"\nНачало обучения: {n_samples} объектов, {n_features} признаков")
        print(f"  Начальная функция потерь: {self.compute_loss(y, np.full_like(y, 0.5), sample_weights):.4f}")
        print("Итерация обучения...")
        
        # Градиентный спуск - основной алгоритм обучения
        for iteration in range(self.n_iters):
            # Прямое распространение
            # Вычисляем линейную комбинацию: z = w1*x1 + w2*x2 + ... + wn*xn + b
            linear_model = np.dot(X, self.weights) + self.bias
            # Применение сигмоиды: ŷ = σ(z) = 1 / (1 + e^(-z))
            # Получение вероятности от 0 до 1
            y_predicted = self.sigmoid(linear_model)
            
            # Вычисление функционала качества
            current_loss = self.compute_loss(y, y_predicted, sample_weights)
            self.loss_history.append(current_loss)

            # Вычисление градиентов из функционала качества
            # Метод градиентного спуска использует градиенты (производные) функции потерь, а не саму функцию.
            dw, db = self.compute_gradients(X, y, y_predicted, sample_weights)
            
            # Обновление весов
            self.weights -= self.lr * dw  # w = w - η * dJ/dw
            self.bias -= self.lr * db  # b = b - η * dJ/db
            
            # Вывод прогресса каждые 200 итераций
            if iteration % 200 == 0 or iteration == self.n_iters - 1:
                predictions = self.predict(X)
                accuracy = np.mean(predictions == y)
                self.accuracy_history.append(accuracy)
                
                grad_norm = np.linalg.norm(dw)
                
                print(f"\n  Итерация {iteration:4d}:")
                print(f"    Loss = {current_loss:.4f}")
                print(f"    Accuracy = {accuracy:.4f}")
                print(f"    ||grad|| = {grad_norm:.6f}")  # это норма (длина) вектора градиента. Она показывает величину градиента. Когда градиент становится очень маленьким (близок к нулю), это означает, что мы приблизились к минимуму функции потерь.

                # Также выводим F1-score для мониторинга
                from sklearn.metrics import f1_score
                f1 = f1_score(y, predictions)
                print(f"    F1 = {f1:.4f}")
                
                # Проверка сходимости
                if grad_norm < 1e-5:
                    print(f"  Градиентный спуск сошелся на итерации {iteration}")
                    break
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей принадлежности к классу 1
        
        Возвращает вероятности от 0 до 1 для каждого объекта
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Предсказание классов (0 или 1)
        
        Параметры:
        X - данные для предсказания
        threshold - порог, выше которого предсказываем класс 1
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_training_history(self):
        """
        Визуализация процесса обучения
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history)
        plt.title('Функция потерь во время обучения')
        plt.xlabel('Итерация')
        plt.ylabel('Log Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history)
        plt.title('Точность во время обучения')
        plt.xlabel('Итерация (каждые 100 шагов)')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Создаем и обучаем модель
print("Создание модели логистической регрессии...")
model = LogisticRegression(learning_rate=0.15, n_iters=2000, class_weight='balanced')

print("Обучение модели на тренировочных данных...")
model.fit(X_train_scaled, y_train)

print("Обучение завершено!")

# =============================================================================
# 2.5 ОЦЕНКА КАЧЕСТВА МОДЕЛИ
# =============================================================================

print("\n=== ОЦЕНКА КАЧЕСТВА МОДЕЛИ ===")

# Функция для оценки модели с разными порогами
def evaluate_with_thresholds(model, X_test, y_test, thresholds):
    """
    оценка модели на разных порогах
        
    Параметры:
        model - тестируемая модель
        X_test - признаки тестовой выборки, по которым предсказываем
        y_test - целевые переменные тестовой выборки
        threshold - порог, выше которого предсказываем класс 1
    """
    results = []
    for threshold in thresholds:
        y_pred = model.predict(X_test, threshold=threshold)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return pd.DataFrame(results)

# Тестируем разные пороги уверенности в 1
thresholds = [0.6, 0.61, 0.65, 0.7]
results_df = evaluate_with_thresholds(model, X_test_scaled, y_test, thresholds)

print("\nСравнение метрик для разных порогов:")
print(results_df.round(4))

# Находим лучший порог по F1-score
best_result = results_df.loc[results_df['f1'].idxmax()]
best_threshold = best_result['threshold']

print(f"\nЛучший порог: {best_threshold:.2f}")
print(f"Метрики при лучшем пороге:")
print(f"  Accuracy:  {best_result['accuracy']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall:    {best_result['recall']:.4f}")
print(f"  F1-Score:  {best_result['f1']:.4f}")

# Финальное предсказание с лучшим порогом
print(f"\nФинальное предсказание с порогом {best_threshold:.2f}")
y_pred_final = model.predict(X_test_scaled, threshold=best_threshold)

# Вычисление финальных метрик
accuracy = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final, zero_division=0)
recall = recall_score(y_test, y_pred_final, zero_division=0)
f1 = f1_score(y_test, y_pred_final, zero_division=0)

print("\n" + "="*60)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
print("="*60)
print(f"Accuracy (Точность):  {accuracy:.4f}")
print(f"Precision (Точность): {precision:.4f}")
print(f"Recall (Полнота):     {recall:.4f}")
print(f"F1-Score (F-мера):    {f1:.4f}")
print("="*60)

# Детальный отчет
print("\nДЕТАЛЬНЫЙ ОТЧЕТ:")
print(classification_report(y_test, y_pred_final))

# Матрица ошибок
print("МАТРИЦА ОШИБОК:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)
print("\nИнтерпретация матрицы ошибок:")
print(f"  True Negative (TN): {cm[0,0]} - правильно предсказанные 0")
print(f"  False Positive (FP): {cm[0,1]} - ложно предсказанные 1")
print(f"  False Negative (FN): {cm[1,0]} - ложно предсказанные 0") 
print(f"  True Positive (TP): {cm[1,1]} - правильно предсказанные 1")

print("\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
print(f"Распределение реальных классов:    {dict(y_test.value_counts())}")
print(f"Распределение предсказанных классов: {dict(pd.Series(y_pred_final).value_counts())}")

model.plot_training_history()