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
    """Busca un archivo en 'modelos/' o en la raíz."""
    rutas = [os.path.join(CARPETA_MODELOS, nombre), nombre]
    for r in rutas:
        if os.path.exists(r):
            return r
    return None


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

    inicio = perf_counter()
    prediccion = int(modelo.predict([pixeles])[0])
    decision = modelo.decision_function([pixeles])[0]
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


if __name__ == "__main__":
    cargar_artefactos()
    print("=" * 60)
    print("DetectIA - Visor Local de Muestras (SVM)")
    print(f"Modelo cargado: {metricas.get('modelo')} | Precisión: {metricas.get('accuracy', 0)*100:.2f}%")
    print("Abre en tu navegador: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
