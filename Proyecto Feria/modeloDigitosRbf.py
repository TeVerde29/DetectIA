import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Carga de datos
df = pd.read_csv('digitos.csv')
X = df.iloc[:, 1:785].values.astype(np.float32) / 255.0
y = df.iloc[:, 0].values

# 2. Partición entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Entrenamiento del modelo (cambia kernel='rbf' y C=5 para mayor precisión)
modelo = SVC(kernel='rbf', C=5, gamma='scale', cache_size=1000, random_state=42)
modelo.fit(X_train, y_train)

# 4. Evaluación
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"Precisión en test: {precision * 100:.2f}%")

# 5. Exportación de artefactos para app.py
os.makedirs('modelos', exist_ok=True)
joblib.dump(modelo, 'modelos/modelo_svm.joblib')

np.savez_compressed(
    'modelos/datos_test.npz',
    X_test=(X_test * 255).astype(np.uint8),
    y_test=y_test.astype(np.int64),
    y_pred=y_pred.astype(np.int64)
)

indices_error = np.where(y_test != y_pred)[0].tolist()
metricas = {
    'modelo': "SVM (kernel='rbf', C=5)",
    'normalizado': True,
    'accuracy': float(precision),
    'aciertos': int(len(y_test) - len(indices_error)),
    'errores': int(len(indices_error)),
    'indices_error': indices_error
}

with open('modelos/metricas_svm.json', 'w', encoding='utf-8') as f:
    json.dump(metricas, f, indent=2)