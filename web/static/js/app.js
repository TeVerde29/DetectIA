const TOTAL_MUESTRAS = (window.DETECTIA_CONFIG && window.DETECTIA_CONFIG.totalMuestras) || 0;
let modeloActivo = "ambos"; // 'ambos', 'svm', 'cnn', etc.
let grosorTrazo = 22;
let modoDibujo = "libre"; // 'libre' o 'grid'
let haDibujado = false;
let gridManual = new Uint8Array(784);
let ultimosPixelesProcesados = null;
let ultimosPixelesExplorer = [];
let indiceActual = 20;

// Metadatos de modelos inyectados desde el backend
const MODELOS_INFO = (window.DETECTIA_CONFIG && window.DETECTIA_CONFIG.modelosInfo) || [];

// --- TEMA CLARO / OSCURO (TEA GREEN) ---
const btnTheme = document.getElementById("btnTheme");
function aplicarTema(oscuro) {
  document.documentElement.setAttribute("data-theme", oscuro ? "dark" : "light");
  btnTheme.textContent = oscuro ? "Tema claro" : "Tema oscuro";
  localStorage.setItem("detectia-theme", oscuro ? "dark" : "light");
  if (modoDibujo === "grid") {
    dibujarGridManualEnCanvas();
  } else {
    recolorearTrazoLibre();
  }
  if (ultimosPixelesProcesados) renderizarGridCanvas(ultimosPixelesProcesados);
  if (ultimosPixelesExplorer.length) renderizarMuestraExplorer(ultimosPixelesExplorer);
}

btnTheme.addEventListener("click", () => {
  const esOscuro = document.documentElement.getAttribute("data-theme") === "dark";
  aplicarTema(!esOscuro);
});

function getInkColor() {
  return getComputedStyle(document.documentElement).getPropertyValue("--canvas-ink").trim() || "#1E2A21";
}

function getGridLineColor() {
  return getComputedStyle(document.documentElement).getPropertyValue("--grid-line").trim() || "rgba(30,42,33,0.1)";
}

// --- SELECTOR DE MODO / MODELO ---
function seleccionarModoModelo(mId) {
  modeloActivo = mId;
  document.querySelectorAll(".model-chip-btn").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-model") === mId);
  });
  // Si ya hay un dibujo o muestra cargada, re-predecir con el nuevo filtro
  if (haDibujado) {
    enviarPrediccionDibujo();
  }
  cargarMuestraExplorer(indiceActual);
}

// --- PESTAÑAS ---
let entrenamientoCargado = false;

function cambiarPestana(p) {
  document.getElementById("tabBtnDraw").classList.toggle("active", p === "draw");
  document.getElementById("tabBtnExplorer").classList.toggle("active", p === "explorer");
  const tabTraining = document.getElementById("tabBtnTraining");
  if (tabTraining) tabTraining.classList.toggle("active", p === "training");

  document.getElementById("viewDraw").classList.toggle("active", p === "draw");
  document.getElementById("viewExplorer").classList.toggle("active", p === "explorer");
  const viewTraining = document.getElementById("viewTraining");
  if (viewTraining) viewTraining.classList.toggle("active", p === "training");

  if (p === "training" && !entrenamientoCargado) {
    cargarDatosEntrenamiento();
  }
}

// --- LIENZO DE DIBUJO ---
const drawCanvas = document.getElementById("drawCanvas");
const drawCtx = drawCanvas.getContext("2d", { willReadFrequently: true });
const gridCanvas = document.getElementById("gridCanvas");
const gridCtx = gridCanvas.getContext("2d");

const inkCanvas = document.createElement("canvas");
inkCanvas.width = 280;
inkCanvas.height = 280;
const inkCtx = inkCanvas.getContext("2d", { willReadFrequently: true });

let isDrawing = false;
let lastX = 0;
let lastY = 0;

function setModoDibujo(nuevoModo) {
  modoDibujo = nuevoModo;
  document.getElementById("modeLibre").classList.toggle("active", nuevoModo === "libre");
  document.getElementById("modeGrid").classList.toggle("active", nuevoModo === "grid");
  document.getElementById("strokeBar").style.display = nuevoModo === "libre" ? "flex" : "none";
  limpiarLienzo();
}

function setGrosor(val, btn) {
  grosorTrazo = val;
  document.querySelectorAll(".stroke-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
}

function limpiarLienzo() {
  inkCtx.clearRect(0, 0, 280, 280);
  drawCtx.clearRect(0, 0, 280, 280);
  gridManual.fill(0);
  ultimosPixelesProcesados = null;
  haDibujado = false;

  if (modoDibujo === "grid") {
    dibujarGridManualEnCanvas();
  }
  limpiarGridCanvas();

  const banner = document.getElementById("drawVerdictBanner");
  banner.className = "verdict-banner idle";
  document.getElementById("drawVerdictText").textContent = "Dibuja un dígito en el lienzo para iniciar la clasificación simultánea.";
  document.getElementById("drawOverallLatency").textContent = "Listo";
  document.getElementById("drawErrorMsg").style.display = "none";
  document.getElementById("drawModelsContainer").innerHTML = "";
}

function limpiarGridCanvas() {
  gridCtx.clearRect(0, 0, 280, 280);
  dibujarLineasCuadricula(gridCtx, 280, 28);
}

function dibujarLineasCuadricula(ctx, size, cells) {
  const step = size / cells;
  ctx.strokeStyle = getGridLineColor();
  ctx.lineWidth = 0.6;
  for (let i = 1; i < cells; i++) {
    ctx.beginPath();
    ctx.moveTo(i * step, 0);
    ctx.lineTo(i * step, size);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i * step);
    ctx.lineTo(size, i * step);
    ctx.stroke();
  }
}

function recolorearTrazoLibre() {
  const imgData = inkCtx.getImageData(0, 0, 280, 280);
  const data = imgData.data;
  const colorHex = getInkColor();
  const r = parseInt(colorHex.slice(1, 3), 16);
  const g = parseInt(colorHex.slice(3, 5), 16);
  const b = parseInt(colorHex.slice(5, 7), 16);

  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] > 0) {
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
    }
  }
  drawCtx.putImageData(imgData, 0, 0);
}

function dibujarGridManualEnCanvas() {
  drawCtx.clearRect(0, 0, 280, 280);
  const colorHex = getInkColor();
  const r = parseInt(colorHex.slice(1, 3), 16);
  const g = parseInt(colorHex.slice(3, 5), 16);
  const b = parseInt(colorHex.slice(5, 7), 16);

  for (let fila = 0; fila < 28; fila++) {
    for (let col = 0; col < 28; col++) {
      const val = gridManual[fila * 28 + col];
      if (val > 0) {
        drawCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${val / 255})`;
        drawCtx.fillRect(col * 10, fila * 10, 10, 10);
      }
    }
  }
  dibujarLineasCuadricula(drawCtx, 280, 28);
}

function pintarCeldaGrid(x, y) {
  const col = Math.floor(x / 10);
  const fila = Math.floor(y / 10);
  if (col < 0 || col >= 28 || fila < 0 || fila >= 28) return;

  const setMax = (f, c, v) => {
    if (f >= 0 && f < 28 && c >= 0 && c < 28) {
      const idx = f * 28 + c;
      if (v > gridManual[idx]) gridManual[idx] = v;
    }
  };
  setMax(fila, col, 255);
  setMax(fila - 1, col, 160);
  setMax(fila + 1, col, 160);
  setMax(fila, col - 1, 160);
  setMax(fila, col + 1, 160);
  dibujarGridManualEnCanvas();
}

function getCoords(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function iniciarTrazo(e) {
  e.preventDefault();
  isDrawing = true;
  const coords = getCoords(e);
  lastX = coords.x;
  lastY = coords.y;
  haDibujado = true;

  if (modoDibujo === "grid") {
    pintarCeldaGrid(coords.x, coords.y);
  } else {
    const ink = getInkColor();
    [inkCtx, drawCtx].forEach(ctx => {
      ctx.fillStyle = ink;
      ctx.beginPath();
      ctx.arc(coords.x, coords.y, grosorTrazo / 2, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

function moverTrazo(e) {
  if (!isDrawing) return;
  e.preventDefault();
  const coords = getCoords(e);
  haDibujado = true;

  if (modoDibujo === "grid") {
    pintarCeldaGrid(coords.x, coords.y);
  } else {
    const ink = getInkColor();
    [inkCtx, drawCtx].forEach(ctx => {
      ctx.strokeStyle = ink;
      ctx.lineWidth = grosorTrazo;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(coords.x, coords.y);
      ctx.stroke();
    });
  }

  lastX = coords.x;
  lastY = coords.y;
}

function finalizarTrazo() {
  if (!isDrawing) return;
  isDrawing = false;
  if (document.getElementById("chkAutoPredict").checked && haDibujado) {
    enviarPrediccionDibujo();
  }
}

drawCanvas.addEventListener("pointerdown", iniciarTrazo);
drawCanvas.addEventListener("pointermove", moverTrazo);
window.addEventListener("pointerup", finalizarTrazo);
window.addEventListener("pointercancel", finalizarTrazo);

document.getElementById("btnClearDraw").addEventListener("click", limpiarLienzo);
document.getElementById("btnPredictDraw").addEventListener("click", enviarPrediccionDibujo);

// --- ENVÍO DE PREDICCIÓN (DIBUJO EN VIVO) ---
async function enviarPrediccionDibujo() {
  const errorDiv = document.getElementById("drawErrorMsg");
  errorDiv.style.display = "none";

  if (!haDibujado) {
    errorDiv.textContent = "El lienzo está vacío. Dibuja un dígito antes de predecir.";
    errorDiv.style.display = "block";
    return;
  }

  let payload = { modelo_id: modeloActivo };
  if (modoDibujo === "grid") {
    const suma = gridManual.reduce((a, b) => a + b, 0);
    if (suma === 0) {
      errorDiv.textContent = "La cuadrícula está vacía.";
      errorDiv.style.display = "block";
      return;
    }
    payload.pixeles = Array.from(gridManual);
    payload.ya_centrado = false;
  } else {
    payload.imagen_base64 = inkCanvas.toDataURL("image/png");
  }

  try {
    const res = await fetch("/api/predecir-dibujo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) {
      errorDiv.textContent = data.error || "Error al predecir.";
      errorDiv.style.display = "block";
      return;
    }

    ultimosPixelesProcesados = data.pixeles_procesados;
    renderizarGridCanvas(data.pixeles_procesados);
    actualizarUIResultadosDibujo(data);
  } catch (err) {
    errorDiv.textContent = "Error de conexión con el servidor.";
    errorDiv.style.display = "block";
  }
}

function actualizarUIResultadosDibujo(data) {
  const modelos = data.modelos || {};
  const keys = Object.keys(modelos);
  const banner = document.getElementById("drawVerdictBanner");
  const verdictText = document.getElementById("drawVerdictText");
  const container = document.getElementById("drawModelsContainer");
  container.innerHTML = "";

  let totalLatency = 0;
  keys.forEach(k => { totalLatency += modelos[k].latencia_ms || 0; });
  document.getElementById("drawOverallLatency").textContent = `Latencia: ${totalLatency.toFixed(1)} ms`;

  if (keys.length >= 2) {
    const m1 = modelos[keys[0]];
    const m2 = modelos[keys[1]];
    if (m1.prediccion === m2.prediccion) {
      banner.className = "verdict-banner agree";
      verdictText.innerHTML = `<strong>Consenso:</strong> Ambos modelos coinciden en el dígito <strong>${m1.prediccion}</strong> (${m1.confianza}% vs ${m2.confianza}% afinidad).`;
    } else {
      banner.className = "verdict-banner disagree";
      verdictText.innerHTML = `<strong>Discrepancia detectada:</strong> ${m1.nombre} predice <strong>${m1.prediccion}</strong> (${m1.confianza}%) mientras que ${m2.nombre} predice <strong>${m2.prediccion}</strong> (${m2.confianza}%).`;
    }
  } else if (keys.length === 1) {
    const m = modelos[keys[0]];
    banner.className = "verdict-banner agree";
    verdictText.innerHTML = `<strong>${m.nombre}:</strong> Dígito predicho <strong>${m.prediccion}</strong> con ${m.confianza}% de confianza.`;
  }

  // Renderizar tarjetas de modelos
  keys.forEach(k => {
    const mod = modelos[k];
    container.appendChild(crearTarjetaModelo(mod, false));
  });
}

function crearTarjetaModelo(mod, mostrarAcierto = false) {
  const card = document.createElement("div");
  card.className = "model-result-card";

  let aciertoBadge = "";
  if (mostrarAcierto && typeof mod.acierto === "boolean") {
    aciertoBadge = mod.acierto
      ? `<span class="badge-verdict correct">✅ Acierto</span>`
      : `<span class="badge-verdict incorrect">❌ Fallo</span>`;
  }

  let juryHtml = "";
  if (Array.isArray(mod.votos_detalle) && mod.votos_detalle.length > 0) {
    const chips = mod.votos_detalle.map(v => {
      const esGanador = v.prediccion === mod.prediccion;
      return `<span class="jury-chip ${esGanador ? "winner" : ""}">${v.nombre}: <strong>${v.prediccion}</strong> (${v.confianza}%)</span>`;
    }).join("");
    juryHtml = `
      <div class="jury-box">
        <div class="jury-header">
          <span>Votación del Jurado</span>
          <span><strong>${mod.votos_ganador}/${mod.total_jurado}</strong> votos</span>
        </div>
        <div class="jury-chips">${chips}</div>
      </div>
    `;
  }

  card.innerHTML = `
    <div class="model-card-header">
      <div>
        <div class="model-card-name">${mod.nombre}</div>
        <div class="model-card-family">${mod.familia || "Clasificador"}</div>
      </div>
      ${aciertoBadge}
    </div>
    <div class="model-big-digit-box">
      <span style="font-size:0.75rem; text-transform:uppercase; color:var(--text-2); font-weight:600;">Predicción</span>
      <div class="model-big-digit">${mod.prediccion}</div>
      <div class="model-meta-row">
        <span>Afinidad: <strong>${mod.confianza}%</strong></span>
        <span>Latencia: <strong>${mod.latencia_ms} ms</strong></span>
      </div>
    </div>
    ${juryHtml}
    <div>
      <div class="affinity-section-title">Distribución de Afinidades (0–9)</div>
      <div class="bars-list" id="bars_${mod.id}"></div>
    </div>
  `;

  // Insertar barras
  setTimeout(() => {
    const bContainer = card.querySelector(`#bars_${mod.id}`);
    if (bContainer && mod.puntajes) {
      renderizarBarrasEnElemento(bContainer, mod.puntajes, mod.prediccion);
    }
  }, 0);

  return card;
}

function renderizarGridCanvas(pixeles) {
  if (!pixeles || pixeles.length !== 784) return;
  gridCtx.clearRect(0, 0, 280, 280);
  const colorHex = getInkColor();
  const r = parseInt(colorHex.slice(1, 3), 16);
  const g = parseInt(colorHex.slice(3, 5), 16);
  const b = parseInt(colorHex.slice(5, 7), 16);

  for (let fila = 0; fila < 28; fila++) {
    for (let col = 0; col < 28; col++) {
      const val = pixeles[fila * 28 + col];
      if (val > 0) {
        gridCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${val / 255})`;
        gridCtx.fillRect(col * 10, fila * 10, 10, 10);
      }
    }
  }
  dibujarLineasCuadricula(gridCtx, 280, 28);
}

function renderizarBarrasEnElemento(container, puntajes, ganador) {
  container.innerHTML = "";
  puntajes.forEach((pct, digito) => {
    const esGanador = digito === ganador;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-digit ${esGanador ? "winner" : ""}">${digito}</span>
      <div class="bar-track">
        <div class="bar-fill ${esGanador ? "winner" : ""}" style="width: ${pct}%"></div>
      </div>
      <span class="bar-pct ${esGanador ? "winner" : ""}">${pct.toFixed(1)}%</span>
    `;
    container.appendChild(row);
  });
}

// --- EXPLORADOR DE DATASET ---
const datasetCanvas = document.getElementById("datasetCanvas");
const datasetCtx = datasetCanvas.getContext("2d");
const inputIndice = document.getElementById("inputIndice");
const explorerErrorMsg = document.getElementById("explorerErrorMsg");
const chkCuadricula = document.getElementById("chkCuadricula");

function renderizarMuestraExplorer(pixeles) {
  if (!pixeles || pixeles.length !== 784) return;
  ultimosPixelesExplorer = pixeles;

  datasetCtx.clearRect(0, 0, 280, 280);
  const colorHex = getInkColor();
  const r = parseInt(colorHex.slice(1, 3), 16);
  const g = parseInt(colorHex.slice(3, 5), 16);
  const b = parseInt(colorHex.slice(5, 7), 16);

  for (let fila = 0; fila < 28; fila++) {
    for (let col = 0; col < 28; col++) {
      const val = pixeles[fila * 28 + col];
      if (val > 0) {
        datasetCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${val / 255})`;
        datasetCtx.fillRect(col * 10, fila * 10, 10, 10);
      }
    }
  }

  if (chkCuadricula.checked) {
    dibujarLineasCuadricula(datasetCtx, 280, 28);
  }
}

async function cargarMuestraExplorer(idx) {
  explorerErrorMsg.style.display = "none";
  indiceActual = idx;
  try {
    const res = await fetch(`/api/muestra/${idx}?modelo=${modeloActivo}`);
    const data = await res.json();
    if (!res.ok) {
      explorerErrorMsg.textContent = data.error || "Error al consultar.";
      explorerErrorMsg.style.display = "inline";
      return;
    }

    inputIndice.value = data.indice;
    document.getElementById("lblIndiceActual").textContent = data.indice;
    document.getElementById("explorerRealVal").textContent = data.valor_real;

    renderizarMuestraExplorer(data.pixeles);
    actualizarUIResultadosExplorer(data);
  } catch (err) {
    explorerErrorMsg.textContent = "Error de conexión con el servidor.";
    explorerErrorMsg.style.display = "inline";
  }
}

function actualizarUIResultadosExplorer(data) {
  const modelos = data.modelos || {};
  const keys = Object.keys(modelos);
  const banner = document.getElementById("explorerVerdictBanner");
  const verdictText = document.getElementById("explorerVerdictText");
  const container = document.getElementById("explorerModelsContainer");
  container.innerHTML = "";

  let totalLatency = 0;
  keys.forEach(k => { totalLatency += modelos[k].latencia_ms || 0; });
  document.getElementById("explorerLatency").textContent = `Latencia: ${totalLatency.toFixed(1)} ms`;

  if (keys.length >= 2) {
    const m1 = modelos[keys[0]];
    const m2 = modelos[keys[1]];
    if (m1.acierto && m2.acierto) {
      banner.className = "verdict-banner agree";
      verdictText.innerHTML = `<strong>Éxito conjunto:</strong> Ambos modelos clasificaron correctamente el dígito <strong>${data.valor_real}</strong>.`;
    } else if (m1.acierto && !m2.acierto) {
      banner.className = "verdict-banner disagree";
      verdictText.innerHTML = `<strong>Discrepancia:</strong> <strong>${m1.nombre}</strong> acertó (${m1.prediccion}), pero <strong>${m2.nombre}</strong> falló (predijo ${m2.prediccion}).`;
    } else if (!m1.acierto && m2.acierto) {
      banner.className = "verdict-banner disagree";
      verdictText.innerHTML = `<strong>Discrepancia:</strong> <strong>${m2.nombre}</strong> acertó (${m2.prediccion}), pero <strong>${m1.nombre}</strong> falló (predijo ${m1.prediccion}).`;
    } else {
      banner.className = "verdict-banner disagree";
      verdictText.innerHTML = `<strong>Ambos modelos difieren del valor real:</strong> Valor real es <strong>${data.valor_real}</strong>. Predicciones: ${m1.prediccion} y ${m2.prediccion}.`;
    }
  } else if (keys.length === 1) {
    const m = modelos[keys[0]];
    if (m.acierto) {
      banner.className = "verdict-banner agree";
      verdictText.innerHTML = `<strong>${m.nombre}:</strong> Clasificación correcta del dígito <strong>${data.valor_real}</strong>.`;
    } else {
      banner.className = "verdict-banner disagree";
      verdictText.innerHTML = `<strong>${m.nombre}:</strong> Clasificación incorrecta. Valor real: <strong>${data.valor_real}</strong>, predicho: <strong>${m.prediccion}</strong>.`;
    }
  }

  keys.forEach(k => {
    const mod = modelos[k];
    container.appendChild(crearTarjetaModelo(mod, true));
  });
}

document.getElementById("formIndice").addEventListener("submit", (e) => {
  e.preventDefault();
  const n = Number(inputIndice.value);
  if (Number.isNaN(n) || n < 0 || n >= TOTAL_MUESTRAS) {
    explorerErrorMsg.textContent = `Índice inválido (0 a ${TOTAL_MUESTRAS - 1}).`;
    explorerErrorMsg.style.display = "inline";
    return;
  }
  cargarMuestraExplorer(n);
});

document.getElementById("btnPrev").addEventListener("click", () => {
  const act = Number(inputIndice.value) || 0;
  cargarMuestraExplorer(Math.max(0, act - 1));
});

document.getElementById("btnNext").addEventListener("click", () => {
  const act = Number(inputIndice.value) || 0;
  cargarMuestraExplorer(Math.min(TOTAL_MUESTRAS - 1, act + 1));
});

document.getElementById("btnAleatorio").addEventListener("click", async () => {
  const res = await fetch("/api/aleatorio");
  const data = await res.json();
  cargarMuestraExplorer(data.indice);
});

document.getElementById("btnSiguienteDiscrepancia").addEventListener("click", async () => {
  const act = Number(inputIndice.value) || 0;
  try {
    const res = await fetch(`/api/siguiente-discrepancia?actual=${act}`);
    const data = await res.json();
    if (res.ok) {
      cargarMuestraExplorer(data.indice);
    } else {
      explorerErrorMsg.textContent = data.error || "No se encontraron más discrepancias.";
      explorerErrorMsg.style.display = "inline";
    }
  } catch (err) {
    explorerErrorMsg.textContent = "Error consultando discrepancia.";
    explorerErrorMsg.style.display = "inline";
  }
});

document.getElementById("btnSiguienteError").addEventListener("click", async () => {
  const act = Number(inputIndice.value) || 0;
  const modeloTarget = modeloActivo === "ambos" ? "svm" : modeloActivo;
  try {
    const res = await fetch(`/api/siguiente-error?modelo=${modeloTarget}&actual=${act}`);
    const data = await res.json();
    if (res.ok) {
      cargarMuestraExplorer(data.indice);
    } else {
      explorerErrorMsg.textContent = data.error || "No hay más errores.";
      explorerErrorMsg.style.display = "inline";
    }
  } catch (err) {
    explorerErrorMsg.textContent = "Error consultando siguiente error.";
    explorerErrorMsg.style.display = "inline";
  }
});

chkCuadricula.addEventListener("change", () => renderizarMuestraExplorer(ultimosPixelesExplorer));

// --- PESTAÑA ÉPOCAS Y GRÁFICAS ---
async function cargarDatosEntrenamiento() {
  entrenamientoCargado = true;
  try {
    const res = await fetch("/api/entrenamiento");
    const data = await res.json();

    const historiales = data.historiales || [];
    const chip = document.getElementById("trainingModelsChip");
    if (historiales.length > 0 && Array.isArray(historiales[0].epocas) && historiales[0].epocas.length > 0) {
      const h = historiales[0];
      const epocas = h.epocas;
      chip.textContent = `${h.modelo || h.model_id} · ${epocas.length} épocas`;
      document.getElementById("epochsEmptyMsg").style.display = "none";
      document.getElementById("epochsGrid").style.display = "grid";

      const labels = epocas.map(e => e.epoca);
      dibujarCurvaEpocas(
        document.getElementById("epochAccCanvas"),
        labels,
        [
          { nombre: "Train Acc", color: "#1a73e8", valores: epocas.map(e => e.train_acc) },
          { nombre: "Val Acc", color: "#1e8e3e", valores: epocas.map(e => e.val_acc) }
        ],
        "%"
      );
      dibujarCurvaEpocas(
        document.getElementById("epochLossCanvas"),
        labels,
        [
          { nombre: "Train Loss", color: "#1a73e8", valores: epocas.map(e => e.train_loss) },
          { nombre: "Val Loss", color: "#d93025", valores: epocas.map(e => e.val_loss) }
        ],
        ""
      );
    } else {
      chip.textContent = "Sin historial CNN aún";
    }

    renderizarBenchmarks(data.benchmarks || []);
    renderizarGaleria(data.galeria || []);
  } catch (err) {
    document.getElementById("trainingModelsChip").textContent = "Error al cargar";
  }
}

function dibujarCurvaEpocas(canvas, labels, series, sufijo) {
  if (!canvas || !labels.length) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const padL = 48, padR = 16, padT = 24, padB = 28;
  ctx.clearRect(0, 0, w, h);

  const allVals = series.flatMap(s => s.valores).filter(v => typeof v === "number");
  let minV = Math.min(...allVals);
  let maxV = Math.max(...allVals);
  if (maxV === minV) { maxV += 1; minV -= 1; }
  const margin = (maxV - minV) * 0.1;
  minV -= margin;
  maxV += margin;

  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const textColor = getComputedStyle(document.documentElement).getPropertyValue("--text-2").trim() || "#444746";
  const gridColor = getComputedStyle(document.documentElement).getPropertyValue("--grid-line").trim() || "rgba(0,0,0,0.08)";

  ctx.strokeStyle = gridColor;
  ctx.fillStyle = textColor;
  ctx.font = "11px Roboto Mono, monospace";
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i) / 4;
    const val = maxV - ((maxV - minV) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(`${val.toFixed(sufijo === "%" ? 1 : 3)}${sufijo}`, 4, y + 4);
  }

  series.forEach((s, sIdx) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    s.valores.forEach((v, idx) => {
      const x = padL + (labels.length === 1 ? plotW / 2 : (plotW * idx) / (labels.length - 1));
      const y = padT + plotH - ((v - minV) / (maxV - minV)) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = s.color;
    ctx.fillText(s.nombre, padL + sIdx * 105, 14);
  });
}

function renderizarBenchmarks(benchmarks) {
  const container = document.getElementById("benchmarksContainer");
  if (!benchmarks.length) {
    container.innerHTML = `<div class="training-empty">Sin métricas registradas todavía.</div>`;
    return;
  }
  const rows = benchmarks.map(b => {
    const m = b.metricas || {};
    const testAcc = m.accuracy ? `${(m.accuracy * 100).toFixed(2)}%` : (b.accuracy_pct ? `${b.accuracy_pct}%` : "—");
    const trainAcc = m.accuracy_train ? `${(m.accuracy_train * 100).toFixed(2)}%` : "—";
    const tiempo = m.tiempo_entrenamiento_seg ? `${m.tiempo_entrenamiento_seg} s` : "—";
    const extraBench = m.benchmarks
      ? Object.entries(m.benchmarks).map(([k, v]) => `${k}: <strong>${v.accuracy_pct}%</strong>`).join(" · ")
      : `${m.aciertos ?? "—"}/${m.muestras_prueba ?? "—"} aciertos`;
    return `
      <tr>
        <td><strong>${b.nombre}</strong></td>
        <td>${trainAcc}</td>
        <td><strong>${testAcc}</strong></td>
        <td>${tiempo}</td>
        <td>${extraBench}</td>
      </tr>
    `;
  }).join("");

  container.innerHTML = `
    <table class="benchmarks-table">
      <thead>
        <tr>
          <th>Modelo</th>
          <th>Train Accuracy</th>
          <th>Test Accuracy</th>
          <th>Tiempo Entrenamiento</th>
          <th>Detalle / Datasets</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderizarGaleria(galeria) {
  const grid = document.getElementById("galleryGrid");
  document.getElementById("galleryCountChip").textContent = `${galeria.length} gráficos`;
  if (!galeria.length) {
    grid.innerHTML = `<div class="training-empty">No se encontraron imágenes de gráficas.</div>`;
    return;
  }
  grid.innerHTML = galeria.map(item => `
    <div class="gallery-card">
      <img src="${item.url}" alt="${item.titulo}" loading="lazy" />
      <div class="gallery-card-title">${item.titulo}</div>
    </div>
  `).join("");
}

// Inicializar tema y vista inicial
const temaGuardado = localStorage.getItem("detectia-theme") === "dark";
aplicarTema(temaGuardado);
limpiarLienzo();
cargarMuestraExplorer(20);
