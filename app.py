# ==============================================================================
# PROYECTO: DetectIA - Servidor Web Local de Demostración (Fase 1 - SVM)
# Permite inspeccionar visualmente cualquier muestra de prueba (0 a 10499)
# y comprobar la predicción del modelo entrenado en tiempo real.
# ==============================================================================

import json
import os
import random
from time import perf_counter

import joblib
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

CARPETA_MODELOS = "modelos"
RUTA_MODELO = os.path.join(CARPETA_MODELOS, "modelo_svm.joblib")
RUTA_TEST = os.path.join(CARPETA_MODELOS, "datos_test.npz")
RUTA_METRICAS = os.path.join(CARPETA_MODELOS, "metricas_svm.json")

modelo = None
X_test = None
y_test = None
metricas = {}


def buscar_archivo(nombre):
    """Busca un archivo en 'modelos/', 'artefactos_detectia_Rapido/modelos/' o en la raíz."""
    rutas = [
        os.path.join(CARPETA_MODELOS, nombre),
        os.path.join("artefactos_detectia_Rapido", "modelos", nombre),
        os.path.join("artefactos_detectia_Rapido", nombre),
        nombre,
    ]
    for r in rutas:
        if os.path.exists(r):
            return r
    return None


def preprocesar_matriz_mnist(pixeles_28x28):
    """
    Aplica el preprocesamiento oficial de MNIST a una matriz 28x28:
    1. Recorte a la caja delimitadora (Bounding Box) del trazo.
    2. Escalado proporcional preservando aspecto dentro de 20x20 px.
    3. Centrado dentro de 28x28 según su Centro de Masa (Center of Mass).
    """
    img = Image.fromarray(pixeles_28x28)
    bbox = img.getbbox()
    if not bbox:
        return np.zeros((28, 28), dtype=np.uint8)

    recorte = img.crop(bbox)
    w, h = recorte.size

    if w > h:
        nuevo_w = 20
        nuevo_h = max(1, int(round((h * 20.0) / w)))
    else:
        nuevo_h = 20
        nuevo_w = max(1, int(round((w * 20.0) / h)))

    recorte_escalado = recorte.resize((nuevo_w, nuevo_h), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (28, 28), 0)
    pos_x = (28 - nuevo_w) // 2
    pos_y = (28 - nuevo_h) // 2
    canvas.paste(recorte_escalado, (pos_x, pos_y))
    matriz = np.array(canvas, dtype=np.float32)

    total_masa = np.sum(matriz)
    if total_masa > 0:
        cy = np.sum(matriz * np.arange(28)[:, None]) / total_masa
        cx = np.sum(matriz * np.arange(28)[None, :]) / total_masa
        desp_x = int(round(14.0 - cx))
        desp_y = int(round(14.0 - cy))
        if desp_x != 0 or desp_y != 0:
            matriz_trasladada = np.roll(matriz, shift=(desp_y, desp_x), axis=(0, 1))
            if desp_y > 0:
                matriz_trasladada[:desp_y, :] = 0
            elif desp_y < 0:
                matriz_trasladada[desp_y:, :] = 0
            if desp_x > 0:
                matriz_trasladada[:, :desp_x] = 0
            elif desp_x < 0:
                matriz_trasladada[:, desp_x:] = 0
            matriz = matriz_trasladada

    return np.clip(matriz, 0, 255).astype(np.uint8)


def cargar_artefactos():
    """Carga en memoria el modelo SVM entrenado, el conjunto de prueba y métricas."""
    global modelo, X_test, y_test, metricas

    ruta_mod = buscar_archivo("modelo_svm.joblib")
    ruta_dat = buscar_archivo("datos_test.npz")
    ruta_met = buscar_archivo("metricas_svm.json")

    if not (ruta_mod and ruta_dat):
        raise FileNotFoundError(
            "No se encontraron los archivos del modelo ('modelo_svm.joblib' y 'datos_test.npz'). "
            "Asegúrate de descomprimir 'artefactos_detectia.zip' en la carpeta 'DetectIA' o en 'modelos/'."
        )

    modelo = joblib.load(ruta_mod)
    datos = np.load(ruta_dat)
    X_test = datos["X_test"]
    y_test = datos["y_test"]

    if ruta_met:
        with open(ruta_met, "r", encoding="utf-8") as f:
            metricas = json.load(f)


def softmax(valores):
    """Convierte puntajes de decisión en porcentajes relativos para visualización."""
    exp_v = np.exp(valores - np.max(valores))
    return (exp_v / exp_v.sum()) * 100.0


@app.route("/")
def index():
    resumen = {
        "modelo": metricas.get("modelo", "SVM (kernel='linear')"),
        "semilla": metricas.get("semilla", 42),
        "total_prueba": int(len(X_test)),
        "accuracy_pct": round(metricas.get("accuracy", 0.0) * 100, 2),
        "aciertos": metricas.get("aciertos", 0),
        "errores": metricas.get("errores", 0),
        "tiempo_train": metricas.get("tiempo_entrenamiento_seg", 0),
    }
    return render_template("index.html", resumen=resumen)


@app.route("/api/muestra/<int:idx>")
def obtener_muestra(idx):
    total = len(X_test)
    if idx < 0 or idx >= total:
        return (
            jsonify({"error": f"El índice debe estar entre 0 y {total - 1}."}),
            400,
        )

    pixeles = X_test[idx]
    valor_real = int(y_test[idx])

    # Adaptación automática si el modelo fue entrenado con pixeles normalizados [0.0, 1.0]
    es_norm = metricas.get("normalizado", False) or getattr(modelo, "_normalizado", False)
    pixeles_eval = (pixeles.astype(np.float32) / 255.0) if es_norm else pixeles

    inicio = perf_counter()
    prediccion = int(modelo.predict([pixeles_eval])[0])
    decision = modelo.decision_function([pixeles_eval])[0]
    latencia_ms = (perf_counter() - inicio) * 1000.0

    puntajes_pct = [round(float(p), 2) for p in softmax(decision)]

    return jsonify(
        {
            "indice": idx,
            "pixeles": pixeles.tolist(),
            "valor_real": valor_real,
            "prediccion": prediccion,
            "acierto": prediccion == valor_real,
            "latencia_ms": round(latencia_ms, 2),
            "puntajes": puntajes_pct,
        }
    )


@app.route("/api/aleatorio")
def muestra_aleatoria():
    idx = random.randint(0, len(X_test) - 1)
    return obtener_muestra(idx)


@app.route("/api/siguiente-error")
def siguiente_error():
    errores = metricas.get("indices_error", [])
    if not errores:
        return jsonify({"error": "No se registraron errores en el conjunto de prueba."}), 404

    actual = request.args.get("actual", default=-1, type=int)
    siguientes = [i for i in errores if i > actual]
    idx_error = siguientes[0] if siguientes else errores[0]
    return obtener_muestra(int(idx_error))


@app.route("/api/predecir-dibujo", methods=["POST"])
def predecir_dibujo():
    datos = request.get_json(silent=True) or {}
    pixeles_raw = datos.get("pixeles")

    if not pixeles_raw or len(pixeles_raw) != 784:
        return jsonify({"error": "Se requiere una matriz de 784 píxeles."}), 400

    matriz_in = np.array(pixeles_raw, dtype=np.uint8).reshape(28, 28)

    if np.sum(matriz_in) == 0:
        return jsonify({"error": "El lienzo está vacío. Por favor, dibuja un número primero."}), 400

    # Preprocesamiento MNIST estándar (caja delimitadora + centro de masa)
    matriz_proc = preprocesar_matriz_mnist(matriz_in)
    pixeles_vector = matriz_proc.reshape(784)

    es_norm = metricas.get("normalizado", False) or getattr(modelo, "_normalizado", False)
    pixeles_eval = (pixeles_vector.astype(np.float32) / 255.0) if es_norm else pixeles_vector

    inicio = perf_counter()
    prediccion = int(modelo.predict([pixeles_eval])[0])
    decision = modelo.decision_function([pixeles_eval])[0]
    latencia_ms = (perf_counter() - inicio) * 1000.0

    puntajes_pct = [round(float(p), 2) for p in softmax(decision)]

    return jsonify(
        {
            "prediccion": prediccion,
            "confianza": puntajes_pct[prediccion],
            "puntajes": puntajes_pct,
            "pixeles_procesados": pixeles_vector.tolist(),
            "latencia_ms": round(latencia_ms, 2),
        }
    )


if __name__ == "__main__":
    cargar_artefactos()
    print("=" * 60)
    print("DetectIA - Visor Local de Muestras (SVM)")
    print(f"Modelo cargado: {metricas.get('modelo')} | Precisión: {metricas.get('accuracy', 0)*100:.2f}%")
    print("Abre en tu navegador: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
