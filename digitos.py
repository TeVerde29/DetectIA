# ==============================================================================
# PROYECTO: DetectIA - Clasificación de Dígitos Manuscritos (MNIST)
# Código extraído directamente del documento 'digitos.docx'
# ==============================================================================

# Manipulación de datos
import pandas as pd
# Operaciones numéricas
import numpy as np
# Creación de gráficos
import matplotlib.pyplot as plt
from time import time
import os

# Herramientas de SVM
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------------------
# 1. Leer el set de datos y consultar información básica
# ------------------------------------------------------------------------------
# Se verifica si 'digitos.csv' está en el directorio actual o dentro de 'DATOS/'
archivo_csv = 'digitos.csv' if os.path.exists('digitos.csv') else os.path.join('DATOS', 'digitos.csv')
print(f"Cargando dataset desde: {archivo_csv} ...")

df = pd.read_csv(archivo_csv)

print("\n--- Información del DataFrame (df.info()) ---")
df.info()

print("\n--- Primeros registros (df.head()) ---")
print(df.head())

print("\n--- Conteo de muestras por dígito (df['label'].value_counts()) ---")
print(df['label'].value_counts())

print("\n--- Generando histograma de etiquetas ---")
df['label'].hist()
plt.title("Distribución de etiquetas (dígitos 0 al 9)")
plt.xlabel("Dígito")
plt.ylabel("Frecuencia")
plt.show()

# ------------------------------------------------------------------------------
# 2. Separación de características (X) y etiquetas (y)
# ------------------------------------------------------------------------------
# Obtener una sección del DataFrame usando los números de columna
pixeles = df.iloc[:, 1:785] # x (784 píxeles: pixel0 a pixel783)
digitos = df.iloc[:, 0:1]   # y (etiqueta del dígito)

# Separar datos para entrenar y para probar
X_train, X_test, y_train, y_test = train_test_split(
    pixeles,
    digitos,
    test_size=0.25
)

# Consultar información de la muestra de entrenamiento
print("\n--- Dimensiones de la muestra de entrenamiento (X_train.shape) ---")
print(X_train.shape)

print("\n--- Primeros registros de entrenamiento (X_train.head()) ---")
print(X_train.head())

# ------------------------------------------------------------------------------
# 3. Función auxiliar para visualizar un dígito
# ------------------------------------------------------------------------------
def mostrar_num(in_data):
    """Genera un gráfico que muestra un registro
    del set de datos.
    Para ello, convierte el array de 1 dimensión en
    una matriz de 28x28.
    """
    matriz = np.array(in_data.values)
    plt.imshow(matriz.reshape(28, 28))
    plt.show()

print("\n--- Visualizando un dígito de prueba (índice 3 de X_test) ---")
mostrar_num(X_test.iloc[3])

# ------------------------------------------------------------------------------
# 4. Creación y entrenamiento del modelo SVM
# ------------------------------------------------------------------------------
# Creación del modelo
modelo = SVC(kernel='linear')

# Entrenamiento
print("\n--- Iniciando entrenamiento del modelo SVM (kernel='linear') ---")
hora_inicio = time()
modelo.fit(X_train.values, y_train.values.ravel())
print('Entrenamiento terminado en {}'.format(time() - hora_inicio))

# ------------------------------------------------------------------------------
# 5. Predicción y evaluación de precisión
# ------------------------------------------------------------------------------
# Crear predicción de datos
print("\n--- Generando predicciones sobre el conjunto de prueba (X_test) ---")
hora_inicio = time()
y_pred = modelo.predict(X_test.values)
print('Predicción terminada en {}'.format(time() - hora_inicio))
print(f"Dimensiones de y_pred: {y_pred.shape}")

precision = accuracy_score(y_test, y_pred)
print(f'Precisión: {precision}')

# ------------------------------------------------------------------------------
# 6. Prueba individual con un caso de prueba
# ------------------------------------------------------------------------------
print("\n--- Prueba individual (registro en índice 1100 de X_test) ---")
prueba = X_test.iloc[1100]
print(f'El resultado era: {y_test.iloc[1100].values[0]}')
mostrar_num(prueba)
prediccion = modelo.predict([prueba.values])
print(f'El dígito es: {prediccion}')

# ------------------------------------------------------------------------------
# 7. Matriz de confusión y visualización
# ------------------------------------------------------------------------------
# Prueba con la matriz de confusión
conf = confusion_matrix(y_test, y_pred)
print("\n--- Matriz de confusión numérica ---")
print(conf)

def plot_cm(cm, classes):
    """Esta función se encarga de generar un gráfico con nuestra
    matriz de confusión.
    cm es la matriz generada por confusion_matrix
    classes es una lista que contiene las posibles clases que puede
    predecir nuestro modelo
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Matriz de confusión')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for indice_fila, fila in enumerate(cm):
        for indice_columna, columna in enumerate(fila):
            if cm[indice_fila, indice_columna] > thresh:
                color = "white"
            else:
                color = "black"
            plt.text(
                indice_columna,
                indice_fila,
                cm[indice_fila, indice_columna],
                color=color,
                horizontalalignment="center"
            )
    plt.ylabel("Valores reales")
    plt.xlabel("Valores calculados")
    plt.tight_layout()
    plt.show()

print("\n--- Mostrando gráfico de Matriz de Confusión ---")
plot_cm(conf, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
