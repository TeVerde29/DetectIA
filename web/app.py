# ==============================================================================
# PROYECTO: DetectIA — Servidor Web Multi-Modelo
# Soporta ejecución individual y comparativa en vivo (SVM vs Red Neuronal)
# ==============================================================================

import base64
import io
import json
import os
import random
from time import perf_counter

import joblib
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_from_directory
from scipy.ndimage import center_of_mass, shift

try:
    import onnxruntime as ort
except ImportError:
    ort = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

CARPETA_MODELOS = os.path.join(ROOT_DIR, "modelos")
RUTA_TEST = os.path.join(CARPETA_MODELOS, "datos_test.npz")

modelos_registrados = {}
X_test = None
y_test = None
indices_discrepancias = []


def preprocesar_imagen_mnist(matriz_gris):
    """
    Aplica el preprocesamiento estándar de MNIST a una matriz de píxeles:
    1. Recorte a la caja delimitadora (Bounding Box) exacta del trazo.
    2. Escalado proporcional con filtro Lanczos dentro de 20x20 px.
    3. Normalización de brillo máximo a 255.
    4. Centrado subpíxel por Centro de Masa (Center of Mass) a (13.5, 13.5).
    """
    img = Image.fromarray(matriz_gris.astype(np.uint8), mode="L")
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

    recorte_escalado = recorte.resize((nuevo_w, nuevo_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    pos_x = (28 - nuevo_w) // 2
    pos_y = (28 - nuevo_h) // 2
    canvas.paste(recorte_escalado, (pos_x, pos_y))
    matriz = np.array(canvas, dtype=np.float32)

    # Estirar contraste para que el trazo tenga intensidad plena
    max_val = np.max(matriz)
    if max_val > 0:
        matriz = (matriz / max_val) * 255.0

    # Centrado subpíxel por Centro de Masa
    if np.sum(matriz) > 0:
        cy, cx = center_of_mass(matriz)
        desp_y = 13.5 - cy
        desp_x = 13.5 - cx
        matriz = shift(matriz, [desp_y, desp_x], order=1, mode="constant", cval=0.0)

    return np.clip(matriz, 0, 255).astype(np.uint8)


def softmax(valores, escala=2.5):
    """Convierte puntajes de decisión en porcentajes relativos."""
    exp_v = np.exp((valores - np.max(valores)) * escala)
    return (exp_v / exp_v.sum()) * 100.0


def cargar_modelos_y_datos():
    """Descubre y carga dinámicamente todos los modelos en subcarpetas de 'modelos/'."""
    global modelos_registrados, X_test, y_test, indices_discrepancias

    # 1. Cargar conjunto de prueba compartido
    if os.path.exists(RUTA_TEST):
        datos = np.load(RUTA_TEST)
        X_test = datos["X_test"]
        y_test = datos["y_test"]
    else:
        raise FileNotFoundError(f"No se encontró el archivo de datos de prueba en '{RUTA_TEST}'.")

    # 2. Descubrir subcarpetas en 'modelos/'
    modelos_registrados = {}
    if os.path.exists(CARPETA_MODELOS):
        subdirs = sorted([d for d in os.listdir(CARPETA_MODELOS) if os.path.isdir(os.path.join(CARPETA_MODELOS, d))])
        for sub in subdirs:
            dir_path = os.path.join(CARPETA_MODELOS, sub)
            info_path = os.path.join(dir_path, "info.json")
            if not os.path.exists(info_path):
                continue

            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)

            tipo = info.get("tipo", "joblib")
            model_id = info.get("id", sub)

            if tipo == "onnx":
                if ort is None:
                    continue
                mod_filename = info.get("archivo_modelo")
                mod_path = os.path.join(dir_path, mod_filename) if mod_filename else ""
                if not mod_path or not os.path.exists(mod_path):
                    continue
                sess = ort.InferenceSession(mod_path, providers=["CPUExecutionProvider"])
                modelos_registrados[model_id] = {
                    "id": model_id,
                    "tipo": "onnx",
                    "dir_path": dir_path,
                    "info": info,
                    "instancia": sess,
                    "input_name": sess.get_inputs()[0].name,
                    "normalizado": info.get("normalizado", True),
                }
            elif tipo == "onnx_ensemble":
                if ort is None:
                    continue
                sesiones = []
                for item in info.get("modelos_onnx", []):
                    fname = item.get("archivo") if isinstance(item, dict) else item
                    m_name = item.get("nombre", fname) if isinstance(item, dict) else fname
                    m_path = os.path.join(dir_path, fname)
                    if os.path.exists(m_path):
                        sess = ort.InferenceSession(m_path, providers=["CPUExecutionProvider"])
                        sesiones.append({
                            "nombre": m_name,
                            "session": sess,
                            "input_name": sess.get_inputs()[0].name,
                        })
                if not sesiones:
                    continue
                modelos_registrados[model_id] = {
                    "id": model_id,
                    "tipo": "onnx_ensemble",
                    "dir_path": dir_path,
                    "info": info,
                    "instancia": sesiones,
                    "normalizado": info.get("normalizado", True),
                }
            else:
                mod_filename = info.get("archivo_modelo")
                mod_path = os.path.join(dir_path, mod_filename) if mod_filename else ""
                if not mod_path or not os.path.exists(mod_path):
                    continue
                instancia = joblib.load(mod_path)
                modelos_registrados[model_id] = {
                    "id": model_id,
                    "tipo": "joblib",
                    "dir_path": dir_path,
                    "info": info,
                    "instancia": instancia,
                    "normalizado": info.get("normalizado", True),
                }

    # 3. Cargar o precalcular índices de discrepancias entre los dos primeros modelos
    disc_cache_path = os.path.join(CARPETA_MODELOS, "discrepancias.json")
    ids = list(modelos_registrados.keys())
    if len(ids) >= 2 and os.path.exists(disc_cache_path):
        try:
            with open(disc_cache_path, "r", encoding="utf-8") as f:
                data_disc = json.load(f)
                if data_disc.get("modelos") == ids[:2]:
                    indices_discrepancias = data_disc.get("indices", [])
        except Exception:
            indices_discrepancias = []

    if len(ids) >= 2 and not indices_discrepancias and X_test is not None:
        p1 = predecir_batch_modelo(modelos_registrados[ids[0]], X_test)
        p2 = predecir_batch_modelo(modelos_registrados[ids[1]], X_test)
        indices_discrepancias = np.where(p1 != p2)[0].tolist()
        try:
            with open(disc_cache_path, "w", encoding="utf-8") as f:
                json.dump({"modelos": ids[:2], "indices": indices_discrepancias, "total": len(indices_discrepancias)}, f)
        except Exception:
            pass


def _resolver_votacion_onnx(sesiones, x_4d):
    """Ejecuta múltiples sesiones ONNX y resuelve Hard Voting + Soft Voting de desempate."""
    votos_detalle = []
    conteo_votos = [0] * 10
    suma_probs = np.zeros(10, dtype=np.float64)

    for s in sesiones:
        logits = s["session"].run(None, {s["input_name"]: x_4d})[0][0]
        probs = softmax(logits, escala=1.0)
        voto = int(np.argmax(probs))
        conteo_votos[voto] += 1
        suma_probs += probs
        votos_detalle.append({
            "nombre": s["nombre"],
            "prediccion": voto,
            "confianza": round(float(probs[voto]), 2),
        })

    probs_promedio = suma_probs / len(sesiones)
    max_votos = max(conteo_votos)
    empatados = [c for c, v in enumerate(conteo_votos) if v == max_votos]
    prediccion = max(empatados, key=lambda c: float(probs_promedio[c]))

    return {
        "prediccion": int(prediccion),
        "puntajes_pct": [round(float(p), 2) for p in probs_promedio],
        "votos_ganador": int(conteo_votos[prediccion]),
        "total_jurado": len(sesiones),
        "conteo_votos": conteo_votos,
        "votos_detalle": votos_detalle,
    }


def predecir_batch_modelo(mod_entry, X_u8, batch_size=512):
    """Predice etiquetas para todo un array uint8 (N, 784) soportando joblib, onnx y onnx_ensemble."""
    tipo = mod_entry.get("tipo", "joblib")
    es_norm = mod_entry.get("normalizado", True)
    X_eval = (X_u8.astype(np.float32) / 255.0) if es_norm else X_u8.astype(np.float32)

    if tipo == "onnx":
        sess = mod_entry["instancia"]
        in_name = mod_entry["input_name"]
        preds = []
        for i in range(0, len(X_eval), batch_size):
            xb = X_eval[i:i + batch_size].reshape(-1, 1, 28, 28)
            logits = sess.run(None, {in_name: xb})[0]
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds)

    if tipo == "onnx_ensemble":
        sesiones = mod_entry["instancia"]
        preds = []
        for i in range(0, len(X_eval), batch_size):
            xb = X_eval[i:i + batch_size].reshape(-1, 1, 28, 28)
            votos_batch = np.zeros((len(xb), 10), dtype=np.int32)
            probs_sum = np.zeros((len(xb), 10), dtype=np.float64)
            for s in sesiones:
                logits = s["session"].run(None, {s["input_name"]: xb})[0]
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
                votos_idx = np.argmax(probs, axis=1)
                votos_batch[np.arange(len(xb)), votos_idx] += 1
                probs_sum += probs
            # Hard voting + soft voting tie-break
            scores = votos_batch.astype(np.float64) + (probs_sum / (len(sesiones) + 1.0))
            preds.append(np.argmax(scores, axis=1))
        return np.concatenate(preds)

    return mod_entry["instancia"].predict(X_eval)


def inferir_con_modelo(mod_entry, pixeles_vector):
    """Ejecuta inferencia con un modelo específico y retorna predicción, puntajes y latencia."""
    tipo = mod_entry.get("tipo", "joblib")
    instancia = mod_entry["instancia"]
    es_norm = mod_entry.get("normalizado", True)
    pixeles_eval = (pixeles_vector.astype(np.float32) / 255.0) if es_norm else pixeles_vector.astype(np.float32)

    inicio = perf_counter()
    extra = {}

    if tipo == "onnx":
        x_4d = pixeles_eval.reshape(1, 1, 28, 28)
        logits = instancia.run(None, {mod_entry["input_name"]: x_4d})[0][0]
        puntajes_pct = [round(float(p), 2) for p in softmax(logits, escala=1.0)]
        prediccion = int(np.argmax(logits))
    elif tipo == "onnx_ensemble":
        x_4d = pixeles_eval.reshape(1, 1, 28, 28)
        res_ens = _resolver_votacion_onnx(instancia, x_4d)
        prediccion = res_ens["prediccion"]
        puntajes_pct = res_ens["puntajes_pct"]
        extra = {
            "votos_ganador": res_ens["votos_ganador"],
            "total_jurado": res_ens["total_jurado"],
            "conteo_votos": res_ens["conteo_votos"],
            "votos_detalle": res_ens["votos_detalle"],
        }
    else:
        prediccion = int(instancia.predict([pixeles_eval])[0])
        if hasattr(instancia, "predict_proba"):
            probs = instancia.predict_proba([pixeles_eval])[0]
            puntajes_pct = [round(float(p * 100.0), 2) for p in probs]
        elif hasattr(instancia, "decision_function"):
            decision = instancia.decision_function([pixeles_eval])[0]
            puntajes_pct = [round(float(p), 2) for p in softmax(decision)]
        else:
            puntajes_pct = [0.0] * 10
            puntajes_pct[prediccion] = 100.0

    latencia_ms = (perf_counter() - inicio) * 1000.0
    confianza = puntajes_pct[prediccion]

    resultado = {
        "id": mod_entry["id"],
        "nombre": mod_entry["info"].get("nombre", mod_entry["id"]),
        "familia": mod_entry["info"].get("familia", ""),
        "tipo": tipo,
        "prediccion": prediccion,
        "confianza": confianza,
        "puntajes": puntajes_pct,
        "latencia_ms": round(latencia_ms, 2),
    }
    resultado.update(extra)
    return resultado


# ==============================================================================
# RUTAS WEB Y ENDPOINTS DE LA API
# ==============================================================================

@app.route("/")
def index():
    lista_modelos = [entry["info"] for entry in modelos_registrados.values()]
    resumen = {
        "total_prueba": int(len(X_test)) if X_test is not None else 0,
        "total_modelos": len(lista_modelos),
        "discrepancias": len(indices_discrepancias),
    }
    return render_template("index.html", modelos=lista_modelos, resumen=resumen)


@app.route("/api/modelos")
def obtener_modelos():
    return jsonify([entry["info"] for entry in modelos_registrados.values()])


@app.route("/api/predecir-dibujo", methods=["POST"])
def predecir_dibujo():
    datos = request.get_json(silent=True) or {}
    imagen_b64 = datos.get("imagen_base64")
    pixeles_raw = datos.get("pixeles")
    ya_centrado = bool(datos.get("ya_centrado", False))
    modelo_seleccionado = datos.get("modelo_id", "ambos")

    if imagen_b64:
        if "," in imagen_b64:
            imagen_b64 = imagen_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(imagen_b64)
        img_pil = Image.open(io.BytesIO(img_bytes))
        if img_pil.mode == "RGBA":
            matriz_in = np.array(img_pil.split()[3], dtype=np.uint8)
        else:
            matriz_in = np.array(img_pil.convert("L"), dtype=np.uint8)
    elif pixeles_raw and len(pixeles_raw) == 784:
        matriz_in = np.array(pixeles_raw, dtype=np.uint8).reshape(28, 28)
    else:
        return jsonify({"error": "Se requiere 'imagen_base64' o matriz de 784 píxeles."}), 400

    if np.sum(matriz_in) == 0:
        return jsonify({"error": "El lienzo está vacío. Por favor, dibuja un número primero."}), 400

    matriz_proc = matriz_in if ya_centrado else preprocesar_imagen_mnist(matriz_in)
    pixeles_vector = matriz_proc.reshape(784)

    # Evaluación multi-modelo o individual
    resultados_modelos = {}
    for m_id, entry in modelos_registrados.items():
        if modelo_seleccionado in ["ambos", "todos", m_id]:
            resultados_modelos[m_id] = inferir_con_modelo(entry, pixeles_vector)

    # Diagnóstico comparativo
    preds = [r["prediccion"] for r in resultados_modelos.values()]
    coinciden = len(set(preds)) == 1 if preds else True
    primer_res = next(iter(resultados_modelos.values())) if resultados_modelos else {}

    return jsonify(
        {
            "modo": modelo_seleccionado,
            "coinciden": coinciden,
            "pixeles_procesados": pixeles_vector.tolist(),
            "modelos": resultados_modelos,
            "prediccion": primer_res.get("prediccion"),
            "confianza": primer_res.get("confianza"),
            "latencia_ms": primer_res.get("latencia_ms"),
            "puntajes": primer_res.get("puntajes", [0.0] * 10),
        }
    )


@app.route("/api/muestra/<int:idx>")
def obtener_muestra(idx):
    total = len(X_test)
    if idx < 0 or idx >= total:
        return jsonify({"error": f"El índice debe estar entre 0 y {total - 1}."}), 400

    pixeles = X_test[idx]
    valor_real = int(y_test[idx])

    modelo_seleccionado = request.args.get("modelo", "ambos")

    resultados_modelos = {}
    for m_id, entry in modelos_registrados.items():
        if modelo_seleccionado in ["ambos", "todos", m_id]:
            res = inferir_con_modelo(entry, pixeles)
            res["acierto"] = res["prediccion"] == valor_real
            resultados_modelos[m_id] = res

    preds = [r["prediccion"] for r in resultados_modelos.values()]
    coinciden = len(set(preds)) == 1 if preds else True
    primer_res = next(iter(resultados_modelos.values())) if resultados_modelos else {}

    return jsonify(
        {
            "indice": idx,
            "pixeles": pixeles.tolist(),
            "valor_real": valor_real,
            "coinciden": coinciden,
            "modelos": resultados_modelos,
            "prediccion": primer_res.get("prediccion"),
            "confianza": primer_res.get("confianza"),
            "acierto": primer_res.get("acierto"),
            "latencia_ms": primer_res.get("latencia_ms"),
            "puntajes": primer_res.get("puntajes", [0.0] * 10),
        }
    )


@app.route("/api/aleatorio")
def muestra_aleatoria():
    idx = random.randint(0, len(X_test) - 1)
    return obtener_muestra(idx)


@app.route("/api/siguiente-discrepancia")
def siguiente_discrepancia():
    """Busca la siguiente muestra donde los modelos discrepan (ideal para demostración universitaria)."""
    if not indices_discrepancias:
        return jsonify({"error": "No se encontraron discrepancias entre los modelos."}), 404

    actual = request.args.get("actual", default=-1, type=int)
    siguientes = [i for i in indices_discrepancias if i > actual]
    idx_disc = siguientes[0] if siguientes else indices_discrepancias[0]
    return obtener_muestra(int(idx_disc))


@app.route("/api/siguiente-error")
def siguiente_error():
    modelo_id = request.args.get("modelo", "svm")
    mod_entry = modelos_registrados.get(modelo_id)
    if not mod_entry:
        return jsonify({"error": f"Modelo '{modelo_id}' no encontrado."}), 404

    # Cargar métricas del modelo si existen
    dir_path = mod_entry.get("dir_path") or os.path.join(CARPETA_MODELOS, modelo_id)
    met_filename = mod_entry["info"].get("archivo_metricas")
    met_path = os.path.join(dir_path, met_filename) if met_filename else None

    errores = None
    if met_path and os.path.exists(met_path):
        with open(met_path, "r", encoding="utf-8") as f:
            metricas = json.load(f)
            if "indices_error" in metricas:
                errores = metricas.get("indices_error", [])

    if errores is None:
        preds = predecir_batch_modelo(mod_entry, X_test)
        errores = np.where(preds != y_test)[0].tolist()

    if not errores:
        return jsonify({"error": "No se registraron errores en este modelo."}), 404

    actual = request.args.get("actual", default=-1, type=int)
    siguientes = [i for i in errores if i > actual]
    idx_error = siguientes[0] if siguientes else errores[0]
    return obtener_muestra(int(idx_error))


@app.route("/api/entrenamiento")
def obtener_datos_entrenamiento():
    """Entrega historiales por época, métricas y galería de gráficas de modelos y repo."""
    historiales = []
    benchmarks = []
    galeria = []

    for m_id, entry in modelos_registrados.items():
        dir_path = entry["dir_path"]
        info = entry["info"]

        hist_file = info.get("archivo_historial", "historial_epocas.json")
        hist_path = os.path.join(dir_path, hist_file)
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                data_h = json.load(f)
                data_h["model_id"] = m_id
                historiales.append(data_h)

        met_file = info.get("archivo_metricas")
        met_path = os.path.join(dir_path, met_file) if met_file else None
        if met_path and os.path.exists(met_path):
            with open(met_path, "r", encoding="utf-8") as f:
                data_m = json.load(f)
                benchmarks.append({
                    "model_id": m_id,
                    "nombre": info.get("nombre", m_id),
                    "accuracy_pct": info.get("accuracy_pct"),
                    "metricas": data_m,
                })

        img_dir = os.path.join(dir_path, "img")
        if os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    galeria.append({
                        "titulo": f"{info.get('nombre', m_id)} — {fname}",
                        "url": f"/modelos-media/{m_id}/img/{fname}",
                        "origen": m_id,
                    })

    repo_img_dir = os.path.join(ROOT_DIR, "img")
    if os.path.isdir(repo_img_dir):
        graficas_repo = [
            "curva_aprendizaje_svc.png",
            "curva_epocas_svm.png",
            "matriz_confusion.png",
            "histograma_distribucion.png",
            "muestra_digitos.png",
            "muestra_individual_1100.png",
        ]
        for fname in graficas_repo:
            if os.path.exists(os.path.join(repo_img_dir, fname)):
                galeria.append({
                    "titulo": f"Fase 1 (SVM) — {fname}",
                    "url": f"/repo-img/{fname}",
                    "origen": "repo",
                })

    return jsonify({
        "historiales": historiales,
        "benchmarks": benchmarks,
        "galeria": galeria,
    })


@app.route("/modelos-media/<model_id>/<path:subpath>")
def servir_media_modelo(model_id, subpath):
    dir_path = os.path.join(CARPETA_MODELOS, model_id)
    if not os.path.isdir(dir_path):
        return jsonify({"error": "Modelo no encontrado."}), 404
    return send_from_directory(dir_path, subpath)


@app.route("/repo-img/<path:filename>")
def servir_img_repo(filename):
    repo_img_dir = os.path.join(ROOT_DIR, "img")
    return send_from_directory(repo_img_dir, filename)


# Inicializar automáticamente al importar o ejecutar
if X_test is None:
    try:
        cargar_modelos_y_datos()
    except Exception as e:
        print(f"Advertencia al inicializar modelos: {e}")

if __name__ == "__main__":
    if X_test is None:
        cargar_modelos_y_datos()
    print("=" * 65)
    print("DetectIA — Servidor Multi-Modelo (Minimalista M3 UI)")
    print(f"Modelos disponibles ({len(modelos_registrados)}):")
    for m_id, entry in modelos_registrados.items():
        info = entry["info"]
        print(f" - [{m_id}] {info.get('nombre')} | Precisión: {info.get('accuracy_pct')}%")
    print(f"Discrepancias detectadas en test set: {len(indices_discrepancias)}")
    print("Abre en tu navegador: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=False)
