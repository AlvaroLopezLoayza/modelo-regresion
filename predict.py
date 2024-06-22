import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# 1. Crear un CSV de ejemplo
# Primero, creamos un pequeño conjunto de datos de ejemplo
data = {
    'departamento': ['HUANUCO', 'LA LIBERTAD', 'JUNIN', 'UCAYALI', 'HUANUCO'],
    'provincia': ['LEONCIO PRADO', 'TRUJILLO', 'CHANCHAMAYO', 'CORONEL PORTILLO', 'LEONCIO PRADO'],
    'distrito': ['RUPA-RUPA', 'EL PORVENIR', 'SAN RAMON', 'YARINACOCHA', 'JOSE CRESPO Y CASTILLO'],
    'enfermedad': ['DENGUE SIN SEÑALES DE ALARMA'] * 5,
    'ano': [2024, 2024, 2024, 2024, 2024],
    'diagnostic': ['A97.0'] * 5,
    'diresa': [10, 13, 12, 25, 10],
    'ubigeo': ['100601', '130102', '120305', '250105', '100604'],
    'edad': [35, 28, 42, 50, 19],
    'tipo_edad': ['A'] * 5,
    'sexo': ['M', 'F', 'M', 'F', 'M']
}

# Crear el DataFrame y guardarlo como CSV
df_new = pd.DataFrame(data)
df_new.to_csv('nuevos_datos.csv', index=False)
print("CSV de ejemplo creado: nuevos_datos.csv")

# 2. Cargar y preparar los datos
df_new = pd.read_csv('nuevos_datos.csv')

# Realizar las mismas transformaciones que hicimos en los datos de entrenamiento
le = LabelEncoder()

# departamento,provincia,distrito,localidad,enfermedad,ano,semana,diagnostic,diresa,ubigeo,localcod,edad,tipo_edad,sexo
categorical_columns = ['departamento', 'provincia', 'distrito', 'localidad', 'enfermedad', 'ano', 'semana', 'diagnostic', 'diresa', 'ubigeo', 'localcod', 'edad', 'tipo_edad', 'sexo']

for col in categorical_columns:
    df_new[col] = le.fit_transform(df_new[col].astype(str))

df_new['sexo'] = df_new['sexo'].map({'M': 0, 'F': 1})

# Asegurarse de que tenemos las mismas columnas que en los datos de entrenamiento
X_new = df_new.drop(['localidad', 'localcod'], axis=1, errors='ignore')

# 3. Cargar el modelo
model = joblib.load('random_forest_model.joblib')

# 4. Realizar predicciones
predictions = model.predict(X_new)

# 5. Mostrar los resultados
df_new['semana_predicha'] = predictions.round().astype(int)
print("\nPredicciones:")
print(df_new[['departamento', 'provincia', 'distrito', 'edad', 'sexo', 'semana_predicha']])

# Guardar los resultados
df_new.to_csv('resultados_predicciones.csv', index=False)
print("\nResultados guardados en: resultados_predicciones.csv")