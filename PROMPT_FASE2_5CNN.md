# Prompt para Continuar DetectIA — Fase 2 (5 CNNs + Big Data + Épocas + Votación en Web)

> Copia y pega el bloque completo de abajo en tu otra cuenta o compártelo con tu compañero para construir todo directamente sobre la estructura actual (`web/`, `modelos/svm/`, `modelos/cnn/`).

---

```markdown
Necesito implementar la Fase 2 de DetectIA sobre la estructura actual del repositorio (`web/`, `modelos/svm/`, `modelos/cnn/`) siguiendo exactamente estas especificaciones técnicas:

### 1. Contexto del entorno y arquitectura
- El entrenamiento pesado debe ir en un **Notebook optimizado para Google Colab (GPU T4)** dentro de `modelos/cnn/DetectIA_Fase2_5CNN_Ensamble_Colab.ipynb` (para no forzar ni calentar la laptop local).
- En local ya tenemos `onnxruntime` instalado (sin PyTorch), así que el servidor web (`web/app.py`) ejecutará las CNNs exportadas a **ONNX** mediante `onnxruntime`.
- **Regla estricta de formato:** Todo el flujo debe mantener exactamente **28 × 28 píxeles** (784 dimensiones, trazo blanco sobre fondo negro, rango `[0.0, 1.0]` en la entrada de los modelos).
- **Primero probar todo en la Web (`web/`)**. No modificar la app Android (`Proyecto Feria/DetectIA/`) hasta validar la web y recibir aprobación.

### 2. Datasets masivos en Colab (Solo fuentes limpias compatibles, ~357.000 muestras)
En el notebook de Colab, descargar y unificar automáticamente con `torchvision.datasets` (sin pedir CSV manuales):
- **MNIST Train (60.000)** (ya incluye los 42k de `digitos.csv` + 18k adicionales).
- **EMNIST Digits Train (240.000)** aplicando corrección obligatoria `.transpose(0, 2, 1)` para corregir la orientación original transpuesta.
- **QMNIST `test50k` (50.000)** muestras extra reconstruidas de NIST SD19.
- **USPS Train (7.291)** dígitos postales reales de 16×16 normalizados a **28 × 28** usando nuestro pipeline estándar (bounding box + Lanczos 20×20 + centrado subpíxel por Centro de Masa en `(13.5, 13.5)`).
- **NO mezclar** SVHN callejero RGB, `pendigits` ni `optdigits` (8×8).
- Aplicar **deduplicación SHA-256** para eliminar repetidos internos y bloquear cualquier muestra presente en los conjuntos de evaluación (`MNIST Test 10k`, `EMNIST Test 40k`, `USPS Test 2k` y opcionalmente `datos_test.npz` si se sube a Colab) para evitar *data leakage*.

### 3. Data Augmentation por época (conservando 28 × 28)
Aplicar *on-the-fly* en entrenamiento:
- Rotación aleatoria `±10°` en ambos sentidos (`degrees=10`).
- Inclinación / cizallamiento (`shear=10`).
- Traslación (`translate=(0.08, 0.08)`) y escala (`scale=(0.90, 1.10)`).
- Variación probabilística suave de grosor de trazo (`MaxPool2d(3, 1, 1)` con `p=0.20`) para simular trazos táctiles gruesos vs lápiz fino.

### 4. Comité de 5 CNNs Heterogéneas + Épocas + Gráficas + Votación
Definir e implementar **5 arquitecturas CNN distintas** en PyTorch (con normalización interna `(x - 0.1307) / 0.3081` dentro de `forward()` para que reciban `[0.0, 1.0]`):
1. `cnn1_tinyvgg`: Kernels estrechos `3×3` profundos + BatchNorm + Dropout.
2. `cnn2_widekernel`: Kernels anchos `7×7` y `5×5` + GELU.
3. `cnn3_resnet`: Bloques residuales con *skip connections* + Global Average Pooling.
4. `cnn4_multiscale`: Bloques paralelos `1×1 + 3×3 + 5×5` tipo Inception.
5. `cnn5_dilated`: Convoluciones dilatadas (*atrous*) + SiLU + Global Average Pooling.

Entrenamiento y monitoreo:
- Entrenar con `AdamW`, `OneCycleLR`, `AMP` (GPU) y `EarlyStopping`, guardando por cada época `train_loss`, `val_loss`, `train_acc`, `val_acc` y `lr`.
- Generar y guardar gráficas PNG:
  - Curvas por época (Loss Train/Val, Accuracy Train/Val, Brecha de Overfitting, Learning Rate).
  - Gráfica comparativa de exactitud (5 CNNs individuales vs Hard Voting vs Soft Voting) y Matriz de Confusión del ensamble.
  - Muestra visual del Data Augmentation `±10°` y orientación corregida.
- **Votación del comité:** Evaluar las 5 CNNs sobre el mismo dígito y decidir el ganador por **mayoría de votos (Hard Voting)** con desempate ponderado mediante promedio Softmax (**Soft Voting**).
- Exportar las 5 redes a **ONNX** (`opset_version=17`, compatibilidad `dynamo=False` si aplica, ejes dinámicos `batch_size`), generar `info.json` (`tipo: "onnx_ensemble"`), `metricas_ensamble.json` y `historial_epocas.json`, y empaquetar todo en **`artefactos_detectia_5cnn.zip`** con descarga automática al terminar (`files.download`).

### 5. Integración en la plataforma Web modular (`web/`)
Actualizar limpiamente los archivos separados actuales sin duplicar archivos en la raíz:
1. **`requirements.txt`:** Asegurar `onnxruntime>=1.18.0`.
2. **`web/app.py`:**
   - Cargar modelos `.onnx` y comités `onnx_ensemble` desde `modelos/` usando `onnxruntime`.
   - Resolver la votación de las 5 CNNs devolviendo `votos_ganador`, `total_jurado`, `conteo_votos` y `votos_detalle` (voto y confianza de cada CNN).
   - Exponer `/api/entrenamiento`, `/modelos-media/<model_id>/<subpath>` y `/repo-img/<filename>`.
   - Hacer compatible `/api/siguiente-error` con modelos ONNX/ensamble.
3. **`web/templates/index.html`, `web/static/css/style.css` y `web/static/js/app.js`:**
   - Actualizar el selector superior a **Comparar Todos** para soportar 2 o más modelos.
   - Mostrar dentro de la tarjeta del ensamble el bloque **Votación del Jurado** (`X/5 votos` + voto individual de cada una de las 5 CNNs).
   - Agregar la pestaña **Épocas y Gráficas** con curvas interactivas en canvas de `historial_epocas.json`, tabla de benchmarks multi-dataset y galería de gráficas exportadas (incluyendo las ya existentes en `img/`, como `curva_aprendizaje_svc.png`, `curva_epocas_svm.png` y `matriz_confusion.png`).
4. Verificar finalmente que `web/app.py` levanta sin errores y que todas las rutas y la pestaña nueva funcionan con 0 errores de JavaScript.
```
