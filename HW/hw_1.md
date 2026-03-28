[YouTube_Shorts_Assignment_1_EDA.ipynb](https://github.com/user-attachments/files/26324390/YouTube_Shorts_Assignment_1_EDA.ipynb)
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "6cff2cd3",
   "metadata": {},
   "source": [
    "# Задание 1 — YouTube Shorts EDA\n",
    "\n",
    "Постановка задачи, выбор метрик и предварительный анализ данных."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "588368a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.express as px\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "\n",
    "df = pd.read_csv(\"YouTube_Shorts_Engagement_and_Growth_Velocity.csv\")\n",
    "df[\"Title_Length\"] = df[\"Title\"].str.len()\n",
    "df[\"High_Growth\"] = (df[\"Views_Per_Day\"] >= df[\"Views_Per_Day\"].quantile(0.75)).astype(int)\n",
    "\n",
    "for c in [\"Views\",\"Likes\",\"Comments\",\"Age_In_Days\",\"Views_Per_Day\",\"Description_Length\",\"Title_Length\"]:\n",
    "    df[f\"log_{c}\"] = np.log1p(df[c])\n",
    "\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9092c8ac",
   "metadata": {},
   "source": [
    "\n",
    "## 1. Постановка задачи\n",
    "\n",
    "**Бизнес-задача:** выявлять Shorts с высокой скоростью роста, чтобы раньше усиливать их продвижение.  \n",
    "**ML-задача:** бинарная классификация — предсказать, относится ли ролик к классу `High_Growth`.  \n",
    "**Целевая переменная:** `High_Growth = 1`, если `Views_Per_Day` находится в верхнем квартиле распределения.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "78a4a688",
   "metadata": {},
   "source": [
    "\n",
    "## 2. Метрики качества\n",
    "\n",
    "**Основная метрика — ROC-AUC.** Она показывает, насколько хорошо модель отделяет быстрорастущие ролики от остальных независимо от порога.  \n",
    "**Дополнительная — F1-score для класса High_Growth.** Она полезна при дисбалансе классов и помогает балансировать precision/recall для поиска перспективных роликов.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "caecb981",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "print(\"Размер:\", df.shape)\n",
    "print(\"Пропуски:\", int(df.isna().sum().sum()))\n",
    "print(\"Дубликаты:\", int(df.duplicated().sum()))\n",
    "print(\"Уникальных каналов:\", df[\"Channel_Name\"].nunique())\n",
    "print(\"Порог High_Growth:\", round(df[\"Views_Per_Day\"].quantile(0.75), 2))\n",
    "print(\"Доля класса High_Growth:\", round(df[\"High_Growth\"].mean(), 4))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15ac8c3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "stats = df[[\"Views\",\"Likes\",\"Comments\",\"Age_In_Days\",\"Engagement_Rate_%\",\"Views_Per_Day\",\"Description_Length\"]].describe().T\n",
    "stats\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d84defc6",
   "metadata": {},
   "source": [
    "## 3. Визуализации seaborn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "209ac3cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "plt.figure(figsize=(8,5))\n",
    "sns.histplot(x=np.log1p(df[\"Views_Per_Day\"]), bins=30, kde=True, color=\"steelblue\")\n",
    "plt.title(\"Распределение log(1 + Views_Per_Day)\")\n",
    "plt.xlabel(\"log(1 + Views_Per_Day)\")\n",
    "plt.ylabel(\"Количество видео\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ba243f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "plt.figure(figsize=(8,5))\n",
    "sns.scatterplot(data=df, x=\"Age_In_Days\", y=\"Views_Per_Day\", hue=\"High_Growth\", alpha=0.7)\n",
    "plt.yscale(\"log\")\n",
    "plt.title(\"Views_Per_Day vs Age_In_Days\")\n",
    "plt.xlabel(\"Возраст видео, дни\")\n",
    "plt.ylabel(\"Views per day (log scale)\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "37dbfe59",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "plt.figure(figsize=(10,7))\n",
    "corr = df[[\"log_Views\",\"log_Likes\",\"log_Comments\",\"log_Age_In_Days\",\"Engagement_Rate_%\",\"log_Views_Per_Day\",\"log_Description_Length\",\"log_Title_Length\"]].corr()\n",
    "sns.heatmap(corr, annot=True, fmt=\".2f\", cmap=\"coolwarm\", center=0)\n",
    "plt.title(\"Корреляции признаков\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2970e1f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "plt.figure(figsize=(8,5))\n",
    "sns.boxplot(data=df, x=\"High_Growth\", y=\"Engagement_Rate_%\", color=\"lightblue\")\n",
    "plt.title(\"Engagement Rate по классам роста\")\n",
    "plt.xlabel(\"High_Growth\")\n",
    "plt.ylabel(\"Engagement Rate %\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bbdfb4a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "top = df[\"Channel_Name\"].value_counts().head(10).sort_values()\n",
    "plt.figure(figsize=(9,6))\n",
    "sns.barplot(x=top.values, y=top.index, color=\"skyblue\")\n",
    "plt.title(\"Топ-10 каналов по числу видео в датасете\")\n",
    "plt.xlabel(\"Количество видео\")\n",
    "plt.ylabel(\"Канал\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb851e2a",
   "metadata": {},
   "source": [
    "## 4. Интерактивная визуализация Plotly"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02486a1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "fig = px.scatter(\n",
    "    df,\n",
    "    x=\"Likes\",\n",
    "    y=\"Views_Per_Day\",\n",
    "    color=\"High_Growth\",\n",
    "    hover_data=[\"Channel_Name\", \"Title\"],\n",
    "    log_x=True,\n",
    "    log_y=True,\n",
    "    title=\"Likes vs Views_Per_Day\"\n",
    ")\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f4756e9c",
   "metadata": {},
   "source": [
    "\n",
    "## 5. Основные выводы\n",
    "\n",
    "1. Распределения просмотров, лайков, комментариев и `Views_Per_Day` сильно асимметричны; полезно применять `log1p`.\n",
    "2. Скорость роста снижается с возрастом видео, поэтому `Age_In_Days` — важный признак.\n",
    "3. `Views` и `Likes` умеренно связаны с `Views_Per_Day`, а `Engagement_Rate_%` не является её полной заменой.\n",
    "4. Датасет не слишком сильно сосредоточен на нескольких каналах, но эффект автора/канала стоит учитывать.\n",
    "5. На следующем этапе имеет смысл сравнить несколько моделей бинарной классификации и провести стратифицированную валидацию.\n"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
