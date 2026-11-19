import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Carga y normalización de datos
print("1. Cargando y normalizando 'digitos.csv'...")
ruta_csv = next(
    (p for p in ['digitos.csv', os.path.join('datos', 'digitos.csv'), os.path.join('..', '..', 'datos', 'digitos.csv')] if os.path.exists(p)),
    'digitos.csv'
)
df = pd.read_csv(ruta_csv)
X = df.iloc[:, 1:785].values.astype(np.float32) / 255.0
y = df.iloc[:, 0].values
print(f"   Dataset listo: {len(X)} muestras.")

# 2. Evaluación rápida (75% / 25%) para métricas y datos_test.npz
print("2. Evaluando precisión y generando datos de prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

modelo_eval = SVC(kernel='linear', cache_size=1000, random_state=42)
modelo_eval.fit(X_train, y_train)
y_pred = modelo_eval.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"   Precisión en test: {precision * 100:.2f}%")

# 3. Entrenamiento final con el 100% de los datos
print("3. Entrenando modelo final SVM Lineal con el 100% de los datos...")
inicio = time.time()
modelo_final = SVC(kernel='linear', cache_size=1000, random_state=42)
modelo_final.fit(X, y)
tiempo_train = time.time() - inicio
print(f"   Entrenamiento completado en {tiempo_train:.2f} segundos.")

# 4. Exportación directa a la carpeta 'modeloslinear/'
carpeta_salida = 'modeloslinear'
print(f"4. Guardando artefactos en '{carpeta_salida}/'...")
os.makedirs(carpeta_salida, exist_ok=True)

joblib.dump(modelo_final, os.path.join(carpeta_salida, 'modelo_svm.joblib'))

np.savez_compressed(
    os.path.join(carpeta_salida, 'datos_test.npz'),
    X_test=(X_test * 255).astype(np.uint8),
    y_test=y_test.astype(np.int64),
    y_pred=y_pred.astype(np.int64)
)

indices_error = np.where(y_test != y_pred)[0].tolist()
metricas = {
    'modelo': "SVM (kernel='linear') [100% datos]",
    'normalizado': True,
    'accuracy': float(precision),
    'tiempo_entrenamiento_seg': round(tiempo_train, 2),
    'aciertos': int(len(y_test) - len(indices_error)),
    'errores': int(len(indices_error)),
    'indices_error': indices_error
}

with open(os.path.join(carpeta_salida, 'metricas_svm.json'), 'w', encoding='utf-8') as f:
    json.dump(metricas, f, indent=2)

print(f"5. ¡Listo! Archivos guardados en '{carpeta_salida}/'.")