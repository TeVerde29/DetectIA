import json
import os
import shutil
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

RUTA_MODELO = os.path.join(BASE_DIR, "modelo_svm.joblib")
RUTA_DATOS_TEST = os.path.join(os.path.dirname(BASE_DIR), "datos_test.npz")
RUTA_JSON_SALIDA = os.path.join(BASE_DIR, "modelo_linear_android.json")
RUTA_ASSETS_ANDROID = os.path.join(
    ROOT_DIR, "Proyecto Feria", "DetectIA", "app", "src", "main", "assets", "modelo_linear_android.json"
)


def predecir_ovo_manual(
    X: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replica exactamente la función de decisión One-vs-One de scikit-learn SVC(kernel='linear'):
      1. Calcula d_k = w_k · x + b_k para los 45 pares (i, j) con i < j.
      2. Si d_k > 0 vota por classes[i]; si d_k <= 0 vota por classes[j].
      3. Calcula puntuación continua estilo _ovr_decision_function de sklearn
         (votes + sum_confidences / (3 * (|sum_confidences| + 1))) para Softmax.
    """
    n_muestras = X.shape[0]
    n_clases = len(classes)

    # Producto matriz: (N, 784) @ (784, 45) + (45,) -> (N, 45)
    decisiones = X @ coef.T + intercept

    votos = np.zeros((n_muestras, n_clases), dtype=np.int32)
    suma_confianzas = np.zeros((n_muestras, n_clases), dtype=np.float64)

    k = 0
    for i in range(n_clases):
        for j in range(i + 1, n_clases):
            d = decisiones[:, k]
            positivo = d > 0.0
            votos[positivo, i] += 1
            votos[~positivo, j] += 1

            suma_confianzas[:, i] += d
            suma_confianzas[:, j] -= d
            k += 1

    # Transformación continua acotada en (-1/3, +1/3) igual que scikit-learn
    confianza_transformada = suma_confianzas / (
        3.0 * (np.abs(suma_confianzas) + 1.0)
    )
    scores_continuos = votos.astype(np.float64) + confianza_transformada

    # Predicción por mayoría de votos (empates resueltos por menor índice, igual que libsvm)
    predicciones = classes[np.argmax(votos, axis=1)]

    # Softmax estable numéricamente sobre los scores continuos
    scores_shift = scores_continuos - np.max(scores_continuos, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shift)
    probabilidades = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return predicciones, votos, probabilidades


def main() -> None:
    if not os.path.exists(RUTA_MODELO):
        raise FileNotFoundError(f"No se encontró el modelo en '{RUTA_MODELO}'.")

    print(f"1. Cargando modelo desde '{RUTA_MODELO}'...")
    modelo = joblib.load(RUTA_MODELO)

    if getattr(modelo, "kernel", None) != "linear":
        raise ValueError(
            f"Se esperaba SVC(kernel='linear'), pero se encontró kernel='{modelo.kernel}'."
        )

    coef = np.asarray(modelo.coef_, dtype=np.float64)
    intercept = np.asarray(modelo.intercept_, dtype=np.float64)
    classes = np.asarray(modelo.classes_, dtype=np.int32)

    print(
        f"   Atributos extraídos: coef_={coef.shape}, intercept_={intercept.shape}, classes_={classes.tolist()}"
    )

    # Redondeo ligero a 6 decimales para reducir tamaño del JSON manteniendo precisión total
    coef_export = np.round(coef, 6)
    intercept_export = np.round(intercept, 6)

    payload = {
        "modelo": "SVC_Linear_OneVsOne",
        "input_size": int(coef.shape[1]),
        "num_classifiers": int(coef.shape[0]),
        "classes": classes.tolist(),
        "intercept": intercept_export.tolist(),
        "coef": coef_export.tolist(),
    }

    print(f"2. Exportando pesos a '{RUTA_JSON_SALIDA}'...")
    with open(RUTA_JSON_SALIDA, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    tamano_kb = os.path.getsize(RUTA_JSON_SALIDA) / 1024.0
    print(f"   Archivo generado: '{RUTA_JSON_SALIDA}' ({tamano_kb:.1f} KB)")

    # Copiar automáticamente a DetectIA/app/src/main/assets/ si el proyecto existe
    carpeta_app_main = os.path.dirname(os.path.dirname(RUTA_ASSETS_ANDROID))
    if os.path.isdir(carpeta_app_main):
        os.makedirs(os.path.dirname(RUTA_ASSETS_ANDROID), exist_ok=True)
        shutil.copy2(RUTA_JSON_SALIDA, RUTA_ASSETS_ANDROID)
        print(f"   Copiado automáticamente a '{RUTA_ASSETS_ANDROID}'.")

    # 3. Verificación rápida: NumPy manual (con pesos exportados) vs modelo.predict()
    print("3. Verificando exactitud matemática (NumPy manual vs modelo.predict())...")
    if os.path.exists(RUTA_DATOS_TEST):
        datos = np.load(RUTA_DATOS_TEST)
        X_val = datos["X_test"].astype(np.float64) / 255.0
        origen = f"{RUTA_DATOS_TEST} ({len(X_val)} muestras)"
    else:
        rng = np.random.default_rng(42)
        X_val = rng.uniform(0.0, 1.0, size=(500, coef.shape[1]))
        origen = "500 muestras sintéticas aleatorias"

    pred_sklearn = modelo.predict(X_val)
    pred_manual, _, probs_manual = predecir_ovo_manual(
        X_val, coef_export, intercept_export, classes
    )

    coincidencias = int(np.sum(pred_sklearn == pred_manual))
    exactitud_replica = (coincidencias / len(X_val)) * 100.0
    print(f"   Dataset de verificación: {origen}")
    print(
        f"   Coincidencias NumPy vs modelo.predict(): {coincidencias}/{len(X_val)} ({exactitud_replica:.4f}%)"
    )

    if coincidencias != len(X_val):
        raise AssertionError(
            "Se detectaron discrepancias entre la predicción manual y modelo.predict()."
        )

    # Ejemplo de confianza Softmax en las primeras 3 muestras
    print("   Muestra de probabilidades Softmax (Top 3) en los primeros ejemplos:")
    for idx in range(min(3, len(X_val))):
        top3 = np.argsort(-probs_manual[idx])[:3]
        resumen = ", ".join(
            f"{int(classes[c])}->{probs_manual[idx, c] * 100:.1f}%" for c in top3
        )
        print(f"     Muestra {idx} (pred={int(pred_manual[idx])}): {resumen}")

    print("4. ¡Verificación 100% exitosa!")


if __name__ == "__main__":
    main()
