# Задание 1 — 30 баллов

## 1. Постановка задачи

### Бизнес-задача
Разработать модель для автоматического определения, является ли SMS-сообщение спамом или нет.

### ML-задача
Задача формулируется как бинарная классификация (spam / ham).

### Данные
Используется датасет SMS Spam Collection, содержащий 5574 текстовых сообщений с метками spam и ham :contentReference[oaicite:0]{index=0}.

---

## 2. Метрика

### Выбранная метрика
F1-score

### Обоснование
Метрика F1-score учитывает баланс между precision и recall, что важно при несбалансированных классах и в задачах обнаружения спама.

---

## 3. EDA (Exploratory Data Analysis)

### Базовый анализ

```python
import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print(df.shape)
print(df['label'].value_counts())
print(df.isnull().sum())
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='label', data=df)
plt.title('Class Distribution')
plt.show()
