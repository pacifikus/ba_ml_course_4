{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Netflix Churn: Сложная модель с подбором гиперпараметров\n",
    "В этом ноутбуке выполняется третье задание:\n",
    "- RandomForest с подбором гиперпараметров\n",
    "- F1-score на тестовой выборке\n",
    "- Глобальная и локальная интерпретация модели через SHAP"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split, RandomizedSearchCV\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.inspection import permutation_importance\n",
    "import shap\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "# Загрузка данных\n",
    "from google.colab import files\n",
    "uploaded = files.upload()\n",
    "df = pd.read_csv('netflix_user_behavior_dataset.csv')\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Целевой признак\n",
    "le_target = LabelEncoder()\n",
    "y = le_target.fit_transform(df['churned'])\n",
    "print(\"Распределение churned:\")\n",
    "print(pd.Series(y).value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Признаки\n",
    "categorical_cols = ['gender', 'country', 'subscription_type', 'payment_method', 'primary_device', 'favorite_genre']\n",
    "numerical_cols = ['age', 'account_age_months', 'monthly_fee', 'devices_used', 'avg_watch_time_minutes',\n",
    "                  'watch_sessions_per_week', 'binge_watch_sessions', 'completion_rate', 'rating_given',\n",
    "                  'content_interactions', 'recommendation_click_rate', 'days_since_last_login']\n",
    "\n",
    "X = df[categorical_cols + numerical_cols].copy()\n",
    "for col in categorical_cols:\n",
    "    le = LabelEncoder()\n",
    "    X[col] = le.fit_transform(X[col])"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Разделение на train/test\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Масштабирование числовых признаков\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = X_train.copy()\n",
    "X_test_scaled = X_test.copy()\n",
    "X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])\n",
    "X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# RandomForest + подбор гиперпараметров\n",
    "param_dist = {\n",
    "    'n_estimators': [100, 200, 300],\n",
    "    'max_depth': [None, 10, 20, 30],\n",
    "    'min_samples_split': [2, 5, 10],\n",
    "    'min_samples_leaf': [1, 2, 4],\n",
    "    'max_features': ['sqrt', 'log2', None]\n",
    "}\n",
    "\n",
    "rf = RandomForestClassifier(random_state=42, class_weight='balanced')\n",
    "rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, scoring='f1', random_state=42, n_jobs=-1)\n",
    "rand_search.fit(X_train_scaled, y_train)\n",
    "\n",
    "best_rf = rand_search.best_estimator_\n",
    "y_pred = best_rf.predict(X_test_scaled)\n",
    "f1 = f1_score(y_test, y_pred)\n",
    "print(\"F1-score RandomForest:\", f1)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Глобальная интерпретация\n",
    "importances = best_rf.feature_importances_\n",
    "indices = np.argsort(importances)[::-1]\n",
    "plt.figure(figsize=(12,6))\n",
    "plt.title(\"Feature Importances\")\n",
    "plt.bar(range(X.shape[1]), importances[indices], align='center')\n",
    "plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Permutation importance\n",
    "perm_importance = permutation_importance(best_rf, X_test_scaled, y_test, n_repeats=10, random_state=42)\n",
    "sorted_idx = perm_importance.importances_mean.argsort()[::-1]\n",
    "plt.figure(figsize=(12,6))\n",
    "plt.bar(range(X.shape[1]), perm_importance.importances_mean[sorted_idx])\n",
    "plt.xticks(range(X.shape[1]), [X.columns[i] for i in sorted_idx], rotation=90)\n",
    "plt.title(\"Permutation Importances\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Локальная интерпретация через SHAP\n",
    "explainer = shap.TreeExplainer(best_rf)\n",
    "shap_values = explainer.shap_values(X_test_scaled)\n",
    "obj_idx = 0\n",
    "shap.initjs()\n",
    "shap.force_plot(explainer.expected_value[1], shap_values[1][obj_idx,:], X_test_scaled.iloc[obj_idx,:])"
   ]
  }
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
 "nbformat": 4,
 "nbformat_minor": 5
}