import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Estrategia 1: Usar un parser más flexible
try:
    df = pd.read_csv('datos.csv', engine='python', on_bad_lines='warn')
except Exception as e:
    print(f"Error al leer el archivo con parser flexible: {e}")
    
    # Estrategia 2: Leer el archivo línea por línea y limpiar
    with open('datos.csv', 'r') as file:
        lines = file.readlines()
    
    cleaned_lines = [line.strip() for line in lines if len(line.split(',')) == 14]
    
    df = pd.read_csv(pd.iotools.StringIO('\n'.join(cleaned_lines)))

# Continuar con la normalización como antes
# Convertir 'ano' y 'semana' a tipo entero
df['ano'] = df['ano'].astype(int)
df['semana'] = df['semana'].astype(int)

# Convertir 'edad' a tipo numérico
df['edad'] = pd.to_numeric(df['edad'], errors='coerce')

# Codificar variables categóricas
le = LabelEncoder()

# departamento,provincia,distrito,localidad,enfermedad,ano,semana,diagnostic,diresa,ubigeo,localcod,edad,tipo_edad,sexo
categorical_columns = ['departamento', 'provincia', 'distrito', 'localidad', 'enfermedad', 'ano', 'semana', 'diagnostic', 'diresa', 'ubigeo', 'localcod', 'edad', 'tipo_edad', 'sexo']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Crear una columna 'tipo_dx' basada en los valores de la columna 'diagnostic'
df['tipo_dx'] = 'C'  # Asumimos que todos son confirmados por defecto

# Codificar 'tipo_dx'
df['tipo_dx'] = le.fit_transform(df['tipo_dx'])

# Codificar 'sexo' como binario
df['sexo'] = df['sexo'].map({'M': 0, 'F': 1})

# Escalar variables numéricas
scaler = StandardScaler()
numeric_columns = ['ano', 'semana', 'edad']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Mostrar las primeras filas del DataFrame normalizado
print(df.head())

# Guardar el DataFrame normalizado
df.to_csv('datos_normalizados.csv', index=False)