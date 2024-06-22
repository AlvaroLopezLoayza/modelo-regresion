import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos normalizados
df = pd.read_csv('datos_normalizados.csv')

# Definir las características (X) y la variable objetivo (y)
X = df.drop(['semana', 'localidad', 'localcod'], axis=1)
y = df['semana']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el espacio de hiperparámetros para la búsqueda
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inicializar el modelo base
rf = RandomForestRegressor(random_state=42)

# Handle missing values using imputation
imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean
X_train_imputed = imputer.fit_transform(X_train)

# Realizar búsqueda aleatoria de hiperparámetros
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                   n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Ajustar el modelo
random_search.fit(X_train_imputed, y_train)

# Obtener el mejor modelo
best_rf = random_search.best_estimator_

# Hacer predicciones en el conjunto de prueba
y_pred = best_rf.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio (MSE): {mse}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse}")
print(f"Error absoluto medio (MAE): {mae}")
print(f"R-cuadrado: {r2}")

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Semanas reales")
plt.ylabel("Semanas predichas")
plt.title("Predicciones vs Valores reales (Random Forest)")
plt.tight_layout()
plt.show()

# Importancia de las características
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': best_rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)

# Visualizar la importancia de las características
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 características más importantes')
plt.tight_layout()
plt.show()

print(feature_importance)

# Guardar el modelo
import joblib
joblib.dump(best_rf, 'random_forest_model.joblib')