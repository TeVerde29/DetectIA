# Documento de Diseño y Arquitectura de Software - DetectIA

## 1. Visión General de la Arquitectura

**DetectIA** implementa una arquitectura desacoplada en dos subsistemas principales:

1. **Subsistema de Inteligencia Artificial y Entrenamiento (Python / Offline Pipeline)**: Se ejecuta una única vez en entorno de desarrollo. Carga el dataset [`DATOS/digitos.csv`](file:///c:/Users/hikih/Documents/Repositorios/DetectIA/DATOS/digitos.csv), entrena la red neuronal, valida métricas y compila el modelo a un formato portable optimizado para la web.
2. **Subsistema de Presentación e Inferencia en Cliente (PWA Móvil / Runtime)**: Aplicación web progresiva basada en HTML5, CSS y Vanilla JS. Se ejecuta directamente en el navegador del teléfono móvil de forma 100% autónoma y offline.

```mermaid
graph LR
    subgraph "1. Entorno de Desarrollo (Python)"
        CSV["DATOS/digitos.csv\n(42.000 muestras)"] --> TRAIN["train.py\n(Preprocesamiento + CNN)"]
        TRAIN --> EVAL["evaluate.py\n(Métricas y Gráficos)"]
        TRAIN --> EXPORT["Exportador a Web\n(model.onnx / weights.json)"]
    end

    subgraph "2. Dispositivo Móvil del Usuario (PWA / Offline)"
        EXPORT -.->|Activo estático empaquetado| ASSET["Modelo de IA en memoria RAM"]
        USER["Usuario (Táctil)"] --> CANVAS["Lienzo de Dibujo"]
        CANVAS --> PREPROC["Preprocesador MNIST\n(Box Crop + Centro de Masa)"]
        PREPROC --> GRID["Visualizador Cuadrícula 28x28"]
        PREPROC --> ENGINE["Motor de Inferencia JS/ONNX"]
        ASSET --> ENGINE
        ENGINE --> RESULT["Predicción (0-9) + Gráfico de Certeza"]
    end
```

---

## 2. Justificación Técnica: ¿Es Necesaria una Base de Datos?

> [!IMPORTANT]
> **Dictamen Técnico: NO es necesaria una Base de Datos.**

A continuación se detalla la justificación técnica y arquitectónica para fundamentar esta decisión ante el jurado o los docentes:

| Criterio | Justificación Técnica |
| :--- | :--- |
| **Naturaleza Stateless (Sin Estado) de la Inferencia** | El reconocimiento de patrones mediante redes neuronales es una función matemática pura: $y = f(X)$, donde $X$ son los 784 píxeles del dibujo y $y$ es la probabilidad de cada dígito. Una vez calculada la salida, el sistema no requiere guardar el dibujo para operar el siguiente. |
| **Persistencia del Conocimiento en el Modelo** | Todo el aprendizaje extraído de las 42.000 imágenes ya queda condensado y "congelado" en la matriz de pesos y sesgos sinápticos del modelo exportado. El archivo del modelo actúa como la "memoria estática" del sistema. |
| **Requisito Crítico de Autonomía 100% Offline** | Incorporar un gestor de base de datos cliente-servidor (como PostgreSQL o MySQL) obligaría al teléfono a tener conexión a una red y a un servidor encendido durante el evento, aumentando puntos de fallo en vivo. |
| **Cero Recolección y Privacidad de Datos** | La aplicación no maneja cuentas de usuario, contraseñas, roles ni transacciones que ameriten tablas relacionales. |
| **Almacenamiento Local Opcional (Sin BD)** | Si en el futuro se deseara guardar los últimos 5 dibujos del usuario como historial visual, el estándar web provee `localStorage` o `IndexedDB` nativo del navegador sin necesidad de levantar ningún motor de bases de datos. |

---

## 3. Arquitectura del Módulo de Machine Learning (Python)

El pipeline de Python se encarga del modelado científico de los datos.

### 3.1. Arquitectura de la Red Neuronal
Para superar el 97% alcanzado por el SVM lineal del documento y tolerar los trazos imperfectos del dedo en pantalla, se implementa una arquitectura neuronal optimizada:

```mermaid
flowchart TD
    IN["Entrada: Matriz 28x28x1 (784 píxeles normalizados [0.0 - 1.0])"]
    C1["Capa Convolucional 1: 32 filtros (3x3), activación ReLU"]
    P1["Max Pooling (2x2) + Dropout(0.25)"]
    C2["Capa Convolucional 2: 64 filtros (3x3), activación ReLU"]
    P2["Max Pooling (2x2) + Dropout(0.25)"]
    FL["Aplanamiento (Flatten)"]
    FC1["Capa Densa (Fully Connected): 128 neuronas + ReLU + Dropout(0.5)"]
    OUT["Capa de Salida Densa: 10 neuronas (Softmax) -> Probabilidades 0 al 9"]

    IN --> C1 --> P1 --> C2 --> P2 --> FL --> FC1 --> OUT
```

### 3.2. Proceso de Exportación
El script de entrenamiento genera:
1. `model.onnx` o `model_weights.json`: Estructura compacta para ser interpretada por el motor de inferencia en JavaScript en el navegador del teléfono.
2. `metricas_entrenamiento.png`: Gráficos de precisión y pérdida por época junto con la matriz de confusión para el informe del proyecto.

---

## 4. Arquitectura del Módulo Cliente Móvil (PWA)

El cliente web sigue la estructura de diseño probada en proyectos como **UNUmbral**:

```mermaid
graph TD
    subgraph UI ["Capa de Interfaz de Usuario"]
        A[Lienzo Canvas Táctil 280x280 px]
        B[Matriz Visualizadora 28x28 celdas]
        C[Panel de Predicción y Probabilidades]
        D[Barra de Herramientas: Limpiar / Reconocer]
    end

    subgraph Logic ["Capa Lógica (app.js)"]
        E[Captura de Trazos Pointer / Touch Events]
        F[Algoritmo de Extracción y Caja Delimitadora]
        G[Algoritmo de Centro de Masa y Normalización]
        H[Motor de Inferencia en Tiempo Real]
    end

    subgraph Service ["Capa de Soporte Offline"]
        I[Service Worker - sw.js]
        J[Cache Storage - Modelo + UI Assets]
    end

    A --> E
    E --> F --> G
    G --> B
    G --> H
    H --> C
    D --> E
    J -.-> Logic
    I -.-> J
```

---

## 5. Algoritmo de Preprocesamiento en Cliente

Este componente es el secreto para que los números dibujados con el dedo sean reconocidos con exactitud:

```mermaid
flowchart TD
    D1["1. Trazo libre en el lienzo grande (ej. 280x280 px)"] --> D2["2. Bounding Box: Detectar los límites mínimos y máximos del trazo (x_min, y_min, x_max, y_max)"]
    D2 --> D3["3. Recorte de la región activa del dígito"]
    D3 --> D4["4. Redimensionamiento proporcional preservando aspecto dentro de una caja de 20x20 píxeles"]
    D4 --> D5["5. Cálculo del Centro de Masa (Centroide) de los píxeles activos"]
    D5 --> D6["6. Pegado y centrado del dígito de 20x20 dentro de la matriz final de 28x28 según su centroide"]
    D6 --> D7["7. Renderizado en el visualizador 28x28 e ingreso inmediato a la IA"]
```

---

## 6. Diagrama de Secuencia de la Interacción del Usuario

```mermaid
sequenceDiagram
    autonumber
    actor Usuario
    participant Canvas as Lienzo Táctil
    participant Preproc as Preprocesador (app.js)
    participant GridUI as Visualizador 28x28
    participant IA as Motor de Inferencia
    participant ResultUI as Panel de Resultados

    Usuario->>Canvas: Dibuja un número con el dedo
    Canvas->>Preproc: Evento touch/pointer finalizado
    Preproc->>Preproc: Calcula Bounding Box y Centro de Masa
    Preproc->>GridUI: Envía matriz 28x28 generada
    GridUI-->>Usuario: Muestra la imagen discretizada en tiempo real
    Preproc->>IA: Envía vector normalizado [784 floats]
    IA->>IA: Ejecuta producto matricial y activación Softmax
    IA->>ResultUI: Retorna dígito ganador y vector de 10 probabilidades
    ResultUI-->>Usuario: Muestra dígito predicho y gráfico de barras animado
```

---

## 7. Estrategia de Empaquetado y Despliegue

1. **Despliegue Web / PWA (GitHub Pages)**:
   * Al ser archivos estáticos (`index.html`, `estilos.css`, `app.js`, `sw.js` y el modelo), se publica directamente en GitHub Pages con HTTPS gratuito.
   * Cuenta con soporte de instalación en Android / iOS (añadir a pantalla de inicio).
2. **Empaquetado Nativo Android (Opcional vía Capacitor)**:
   * Mediante `capacitor.config.json`, el proyecto puede compilarse directamente a un archivo `.apk` instalable en teléfonos Android usando Android Studio.
3. **Servidor Local de Demostración (Python Flask)**:
   * Se incluye un archivo `server.py` que permite ejecutar `python server.py` para levantar un servidor local en caso de que la cátedra solicite una demostración en vivo de arquitectura cliente-servidor.
