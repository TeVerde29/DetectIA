# ==============================================================================
# PROYECTO: DetectIA - Clasificación de Dígitos Manuscritos (MNIST)
# Baseline Fase 1: Clasificador SVM Lineal con Semilla Reproducible
# ==============================================================================

import json
import os
import sys
from time import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Usar backend sin ventana por defecto para no congelar la terminal;
# pasar --gui si se desean abrir las ventanas interactivas de Matplotlib.
MODO_GUI = "--gui" in sys.argv
if not MODO_GUI:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

SEMILLA = 42
CARPETA_IMG = "img"
CARPETA_MODELOS = "modelos"
os.makedirs(CARPETA_IMG, exist_ok=True)
os.makedirs(CARPETA_MODELOS, exist_ok=True)


def localizar_dataset():
    """Localiza digitos.csv tolerando rutas en minúsculas o mayúsculas."""
    rutas_posibles = [
        "digitos.csv",
        os.path.join("datos", "digitos.csv"),
        os.path.join("DATOS", "digitos.csv"),
        os.path.join("..", "..", "datos", "digitos.csv"),
        os.path.join("..", "..", "DATOS", "digitos.csv"),
    ]
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(
        "No se encontró 'digitos.csv' en el directorio actual ni en 'datos/'."
    )


def mostrar_num(in_data, titulo="Dígito 28x28", guardar_en=None):
    """Convierte un vector de 784 píxeles en matriz 28x28 y lo renderiza."""
    matriz = np.array(in_data.values if hasattr(in_data, "values") else in_data)
    plt.figure(figsize=(4, 4))
    plt.imshow(matriz.reshape(28, 28), cmap="gray")
    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()
    if guardar_en:
        plt.savefig(guardar_en, dpi=150)
    if MODO_GUI:
        plt.show()
    plt.close()


def plot_cm(cm, classes, guardar_en=None):
    """Genera y guarda el mapa de calor de la matriz de confusión."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Matriz de confusión — SVM Lineal (semilla=42)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for indice_fila in range(cm.shape[0]):
        for indice_columna in range(cm.shape[1]):
            valor = cm[indice_fila, indice_columna]
            color = "white" if valor > thresh else "black"
            plt.text(
                indice_columna,
                indice_fila,
                valor,
                color=color,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=9,
            )
    plt.ylabel("Valores reales")
    plt.xlabel("Valores predichos")
    plt.tight_layout()
    if guardar_en:
        plt.savefig(guardar_en, dpi=150)
    if MODO_GUI:
        plt.show()
    plt.close()


def main():
    # --------------------------------------------------------------------------
    # 1. Leer el set de datos y consultar información básica
    # --------------------------------------------------------------------------
    archivo_csv = localizar_dataset()
    print(f"Cargando dataset desde: {archivo_csv} ...")
    df = pd.read_csv(archivo_csv)

    print("\n--- Información del DataFrame ---")
    df.info()

    print("\n--- Primeros 5 registros ---")
    print(df.head())

    print("\n--- Conteo de muestras por dígito ---")
    conteo = df["label"].value_counts().sort_index()
    print(conteo)

    plt.figure(figsize=(8, 4))
    conteo.plot(kind="bar", color="#2563eb", edgecolor="black")
    plt.title("Distribución de etiquetas (dígitos 0 al 9)")
    plt.xlabel("Dígito")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMG, "histograma_distribucion.png"), dpi=150)
    if MODO_GUI:
        plt.show()
    plt.close()

    # --------------------------------------------------------------------------
    # 2. Separación de características (X) y etiquetas (y) con semilla fija
    # --------------------------------------------------------------------------
    pixeles = df.iloc[:, 1:785]
    digitos = df.iloc[:, 0:1]

    X_train, X_test, y_train, y_test = train_test_split(
        pixeles,
        digitos,
        test_size=0.25,
        random_state=SEMILLA,
    )

    print(f"\n--- Partición reproducible (random_state={SEMILLA}) ---")
    print(f"Entrenamiento: {X_train.shape[0]} muestras x {X_train.shape[1]} píxeles")
    print(f"Prueba:        {X_test.shape[0]} muestras x {X_test.shape[1]} píxeles")

    # Visualizar una muestra inicial del conjunto de prueba
    mostrar_num(
        X_test.iloc[3],
        titulo=f"Muestra de prueba #3 (Real: {y_test.iloc[3].values[0]})",
        guardar_en=os.path.join(CARPETA_IMG, "muestra_prueba_3.png"),
    )

    # --------------------------------------------------------------------------
    # 3. Creación y entrenamiento del modelo SVM Lineal
    # --------------------------------------------------------------------------
    modelo = SVC(kernel="linear", random_state=SEMILLA)

    print("\n--- Iniciando entrenamiento del modelo SVM (kernel='linear') ---")
    inicio_train = time()
    modelo.fit(X_train.values, y_train.values.ravel())
    tiempo_train = time() - inicio_train
    print(f"Entrenamiento terminado en {tiempo_train:.2f} segundos")

    # --------------------------------------------------------------------------
    # 4. Predicción y evaluación de precisión
    # --------------------------------------------------------------------------
    print("\n--- Evaluando sobre las 10.500 muestras de prueba ---")
    inicio_pred = time()
    y_pred = modelo.predict(X_test.values)
    tiempo_pred = time() - inicio_pred
    print(f"Predicción terminada en {tiempo_pred:.2f} segundos")

    y_test_arr = y_test.values.ravel()
    precision = accuracy_score(y_test_arr, y_pred)
    print(f"\n>>> Precisión global (Accuracy): {precision:.5f} ({precision * 100:.2f}%)")

    print("\n--- Reporte de Clasificación por Dígito ---")
    print(classification_report(y_test_arr, y_pred, digits=4))

    # --------------------------------------------------------------------------
    # 5. Prueba individual (registro 1100 de X_test)
    # --------------------------------------------------------------------------
    idx_demo = 1100
    real_demo = int(y_test_arr[idx_demo])
    pred_demo = int(modelo.predict([X_test.iloc[idx_demo].values])[0])
    print(f"--- Prueba individual (índice {idx_demo}) ---")
    print(f"Valor real: {real_demo} | Predicción SVM: {pred_demo}")
    mostrar_num(
        X_test.iloc[idx_demo],
        titulo=f"Índice {idx_demo} | Real: {real_demo} | Predicción: {pred_demo}",
        guardar_en=os.path.join(CARPETA_IMG, "muestra_individual_1100.png"),
    )

    # --------------------------------------------------------------------------
    # 6. Matriz de confusión
    # --------------------------------------------------------------------------
    conf = confusion_matrix(y_test_arr, y_pred)
    print("\n--- Matriz de confusión numérica ---")
    print(conf)
    plot_cm(
        conf,
        list(range(10)),
        guardar_en=os.path.join(CARPETA_IMG, "matriz_confusion.png"),
    )

    # --------------------------------------------------------------------------
    # 7. Exportar modelo entrenado, datos de prueba y métricas para la interfaz web
    # --------------------------------------------------------------------------
    ruta_modelo = os.path.join(CARPETA_MODELOS, "modelo_svm.joblib")
    ruta_test = os.path.join(CARPETA_MODELOS, "datos_test.npz")
    ruta_metricas = os.path.join(CARPETA_MODELOS, "metricas_svm.json")

    joblib.dump(modelo, ruta_modelo)
    np.savez_compressed(
        ruta_test,
        X_test=X_test.values.astype(np.uint8),
        y_test=y_test_arr.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
    )

    indices_error = np.where(y_test_arr != y_pred)[0].tolist()
    metricas = {
        "modelo": "SVM (kernel='linear')",
        "semilla": SEMILLA,
        "total_muestras": int(len(df)),
        "muestras_entrenamiento": int(len(X_train)),
        "muestras_prueba": int(len(X_test)),
        "accuracy": float(precision),
        "tiempo_entrenamiento_seg": round(tiempo_train, 2),
        "tiempo_prediccion_seg": round(tiempo_pred, 2),
        "aciertos": int(len(y_test_arr) - len(indices_error)),
        "errores": int(len(indices_error)),
        "indices_error": indices_error,
    }
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"\nArtefactos guardados correctamente en '{CARPETA_MODELOS}/':")
    print(f" - Modelo entrenado : {ruta_modelo}")
    print(f" - Datos de prueba  : {ruta_test}")
    print(f" - Métricas         : {ruta_metricas}")


if __name__ == "__main__":
    main()
