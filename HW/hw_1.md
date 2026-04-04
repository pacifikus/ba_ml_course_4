import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 1. Постановка задачи
# ======================
# Бизнес: Предсказать, какие пользователи Netflix с высокой вероятностью отпадут, чтобы вовремя их удержать.
# ML: Бинарная классификация (целевой столбец 'churned')
# Данные: демография пользователя, подписка, поведение и взаимодействие с контентом.

# ======================
# 2. Загрузка данных
# ======================
file_path = 'netflix_user_behavior_dataset.csv'
df = pd.read_csv(file_path)
print(df.head())

# Преобразуем целевой столбец в категорию для корректной работы Seaborn
df['churned'] = df['churned'].astype('category')

# ======================
# 3. Выбор метрики
# ======================
# Метрика: F1-score, так как классы могут быть несбалансированными (churned vs not churned).
# F1-score учитывает точность и полноту, что важно для бизнес-задачи удержания пользователей.

# ======================
# 4. EDA
# ======================

# 4.1 Основные характеристики
print("Общая информация о данных:")
df.info()
print("\nСтатистики по числовым признакам:")
print(df.describe())

# 4.2 Проверка распределения целевого признака
plt.figure(figsize=(6,4))
sns.countplot(x='churned', data=df)
plt.title('Распределение пользователей: отток vs нет оттока')
plt.show()

# 4.3 Распределение пользователей по возрасту и полу
plt.figure(figsize=(8,5))
sns.histplot(df, x='age', bins=30, kde=True, hue='churned')
plt.title('Распределение возраста пользователей с учетом оттока')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='gender', hue='churned', data=df)
plt.title('Пол пользователей и отток')
plt.show()

# 4.4 Распределение по странам
top_countries = df['country'].value_counts().nlargest(10).index
plt.figure(figsize=(10,6))
sns.countplot(y='country', hue='churned', data=df[df['country'].isin(top_countries)])
plt.title('Отток по топ-10 стран')
plt.show()

# 4.5 Влияние активности на отток
activity_features = ['avg_watch_time_minutes', 'watch_sessions_per_week', 'binge_watch_sessions', 'completion_rate', 'recommendation_click_rate', 'days_since_last_login']

plt.figure(figsize=(12,8))
sns.boxplot(x='churned', y='avg_watch_time_minutes', data=df)
plt.title('Среднее время просмотра и отток')
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(x='churned', y='watch_sessions_per_week', data=df)
plt.title('Количество сессий просмотра в неделю и отток')
plt.show()

# 4.6 Корреляции числовых признаков
plt.figure(figsize=(12,10))
corr = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Корреляции числовых признаков')
plt.show()

# ======================
# Комментарии по EDA
# - Пользователи с низкой активностью (меньше сессий и минут просмотра) чаще churn.
# - Пол и возраст показывают небольшое влияние, страны с разной подпиской имеют различия в оттоке.
# - В будущем стоит обратить внимание на нормализацию числовых признаков и кодирование категориальных для ML.
# - Пропусков нет, данные чистые.
