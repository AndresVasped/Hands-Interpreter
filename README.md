# 🤟 Hands Interpreter: Reconocimiento de Lengua de Señas en Tiempo Real

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-blueviolet.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![PyTest](https://img.shields.io/badge/PyTest-passing-brightgreen.svg)
![Semgrep](https://img.shields.io/badge/Semgrep-Secured-blue.svg)
![LoadTesting](https://img.shields.io/badge/Latency-3.14ms-yellow.svg)

**HOLA**
https://github.com/user-attachments/assets/bdd94047-a504-436d-b7b2-a99418691a73

**ABCD**
https://github.com/user-attachments/assets/016cb20e-370e-4ca3-b722-e7a9b3cb5f61

**Hands Interpreter** es un intérprete de lengua de señas en tiempo real construido con Python, Visión por Computador y Deep Learning. Utiliza la cámara del computador para reconocer siete letras estáticas del alfabeto manual (**A, B, C, D, H, L y O**) mediante **MediaPipe Hands** y una **Red Neuronal Convolucional 1D (CNN 1D)**.

### 🌍 Contexto e Impacto
Según la Federación Mundial de Sordos, existen más de 70 millones de personas sordas en el mundo que utilizan más de 300 lenguas de señas (ONU). En Colombia, la Lengua de Señas Colombiana (LSC) está reconocida oficialmente por la **Ley 324 de 1996**. Este proyecto permite:
- **Reconocer gestos manuales en tiempo real** mostrando la predicción sobre el video.
- **Servir como base educativa** para el aprendizaje y la comunicación inclusiva.
- **Escalar progresivamente** hacia el alfabeto completo, señas dinámicas y síntesis de voz.

---

## 🏗️ Arquitectura de Software y Pipeline de ML

El flujo de datos opera de forma local y en tiempo real:
`Cámara (OpenCV)` → `MediaPipe Hands (21 landmarks)` → `Vector (63 valores x,y,z)` → `StandardScaler (scaler.pkl)` → `CNN 1D (hand_model.keras)` → `Letra en pantalla`

### Capas del Sistema
| Capa | Responsabilidad |
| :--- | :--- |
| **Visual** | Captura el flujo de video con OpenCV y entrega *frames* de 800×600. |
| **Percepción** | MediaPipe Hands detecta la mano y extrae 21 puntos de referencia 3D `(x, y, z)`. |
| **Datos** | Convierte los puntos en un vector de 63 valores, los normaliza con `StandardScaler` y codifica las etiquetas (*one-hot*). |
| **Modelo** | Red Convolucional 1D con conexión residual que clasifica los 63 valores en probabilidades por letra. |
| **Despliegue** | Carga `hand_model.keras` y `scaler.pkl` en memoria, predice en vivo y renderiza el resultado en pantalla. |

### Patrones de Diseño
Para garantizar escalabilidad y código limpio, la arquitectura se apoya en tres patrones de diseño:
1. **Strategy Pattern:** Permite intercambiar algoritmos de reconocimiento (letras estáticas actuales, futuros números o gestos dinámicos) sin alterar el motor principal.
2. **Singleton Pattern:** Aplicado al gestor de cámara (`Camera`), asegurando una única instancia activa del hardware de video para evitar bloqueos y fugas de memoria.
3. **Factory Pattern:** Centraliza la instanciación de módulos interconectados (interfaz, detección y predicción).

---

## 📊 Selección, Preparación de Datos y EDA

En lugar de depender de datasets públicos con fondos controlados, construimos un **dataset propio** capturado con webcam (`saveDataset.py`) almacenado en `data/dataset/<LETRA>/`, con un total de **3.860 imágenes**.

Trabajar con coordenadas 3D (*landmarks*) en vez de imágenes crudas reduce la dependencia del fondo y la iluminación, y disminuye drásticamente el costo computacional frente a una CNN 2D tradicional.

### Distribución del Dataset (Filtro MediaPipe con confianza 0.7)
| Letra | Imágenes Totales | Muestras Detectadas (aprox.) | % Aprovechado |
| :---: | :---: | :---: | :---: |
| **A** | 585 | 218 | 37 % |
| **B** | 598 | 597 | 100 % |
| **C** | 407 | 159 | 39 % |
| **D** | 604 | 431 | 71 % |
| **H** | 591 | 551 | 93 % |
| **L** | 293 | 279 | 95 % |
| **O** | 782 | 303 | 39 % |
| **Total** | **3.860** | **≈ 2.540** | **≈ 66 %** |

<img width="481" height="242" alt="image" src="https://github.com/user-attachments/assets/406abf6a-1f9b-4371-bd09-8a177d71d3c3" />


**Hallazgos del Análisis Exploratorio (EDA):**
- **Desbalance por geometría:** `B` y `H` concentran la mayoría de muestras y `C` es la clase minoritaria. El desbalance proviene del filtro de MediaPipe: aunque `O` tiene más fotos, solo se aprovecha el 39 %.
- **Pérdida en manos cerradas:** Las letras con dedos curvos o puño cerrado (`A`, `C`, `O`) presentan mayor dificultad de detección de *landmarks*.
- **Calidad de datos:** Existen 53 imágenes duplicadas y *frames* consecutivos muy similares, lo que representa un riesgo de memorización si no se controla en futuras versiones.

### Preprocesamiento y Aumento de Datos (*Data Augmentation*)
- **Codificación y Normalización:** Los 63 valores `(x, y, z)` se estandarizan con `StandardScaler` (media 0, desviación 1) y las etiquetas se convierten a formato *one-hot*.
- **División:** 80 % entrenamiento / 20 % prueba, estratificada por clase (`random_state=42`).
- **Aumento de Datos:** Sobre el conjunto de entrenamiento se generan dos copias adicionales: una con **ruido gaussiano ($\sigma = 0.03$)** para simular variaciones de detección y otra con **escalado aleatorio ($0.9\text{ a }1.1$)** para simular distintas distancias a la cámara. Esto amplía el entrenamiento de **~2.030 a ~6.090 muestras**.

---

## 🧠 Diseño e Implementación del Modelo (CNN 1D)

Evolucionamos la red densa inicial (`63 → 128 → 64 → 7`) hacia una **Red Neuronal Convolucional 1D con Conexión Residual**. Al reestructurar los 63 valores en una secuencia de `21 × 3` (siguiendo el orden anatómico de MediaPipe: muñeca y dedos de base a punta), los filtros convolucionales capturan patrones locales entre articulaciones vecinas (ej. dedo doblado vs. estirado).

| Bloque | Capas | Forma de Salida |
| :--- | :--- | :---: |
| **Entrada** | `Input(63)` → `Reshape((21, 3))` | `21 × 3` |
| **Convolución 1** | `Conv1D(64, k=3, ReLU)` → `BatchNormalization` → `MaxPooling1D(2)` → `Dropout(0.25)` | `10 × 64` |
| **Bloque Residual** | `Conv1D(128, k=3, ReLU)` → `BatchNormalization` → `Conv1D(128, k=3)` + Atajo `Conv1D(128, k=1)` → `ReLU` | `10 × 128` |
| **Clasificador** | `GlobalAveragePooling1D` → `Dense(64, ReLU)` → `BatchNormalization` → `Dropout(0.3)` → `Dense(7, Softmax)` | `7` |

- **Parámetros totales:** 92.679 (92.167 entrenables).
- **Estrategia de entrenamiento:** Optimizador `Adam`, pérdida `categorical_crossentropy`, lotes de 32, hasta 100 épocas con `EarlyStopping` (paciencia de 15 épocas restaurando mejores pesos) y `ReduceLROnPlateau` (factor 0.5 tras 10 épocas sin mejora). El entrenamiento convergió y se detuvo en la **época 35**.

<img width="700" height="236" alt="image" src="https://github.com/user-attachments/assets/f8982232-d0d0-489a-9257-816a5f426ac4" />

---

## 📈 Evaluación y Análisis de Resultados

El modelo alcanzó **506 aciertos de 508 muestras de prueba**, logrando una **exactitud global del 99.61 %**.

<img width="744" height="267" alt="image" src="https://github.com/user-attachments/assets/03557316-d0bd-4b3c-8d08-9aa92531e14a" />


| Letra | Precisión | Recall | F1-Score | Muestras |
| :---: | :---: | :---: | :---: | :---: |
| **A** | 0.98 | 1.00 | 0.99 | 44 |
| **B** | 1.00 | 1.00 | 1.00 | 119 |
| **C** | 0.97 | 1.00 | 0.98 | 32 |
| **D** | 1.00 | 0.99 | 0.99 | 86 |
| **H** | 1.00 | 1.00 | 1.00 | 110 |
| **L** | 1.00 | 1.00 | 1.00 | 56 |
| **O** | 1.00 | 0.98 | 0.99 | 61 |
| **Promedio Macro** | **0.99** | **1.00** | **0.99** | **508** |

### Análisis de Errores y Lectura Crítica
- **Confusiones detectadas:** Solo se registraron **2 errores**: una `D` predicha como `C` y una `O` predicha como `A`. Ambos pares comparten geometría similar (dedos curvos alrededor del pulgar o mano cerrada). Todas las clases lograron un **AUC de 1.00**.
- **Limitaciones metodológicas identificadas:** El 99.61 % refleja el desempeño sobre condiciones similares al dataset original, pero presenta sesgos de evaluación a corregir en la siguiente iteración:
  1. El conjunto de prueba se utilizó también como validación para los *callbacks* (`EarlyStopping` / `ReduceLROnPlateau`).
  2. `StandardScaler` se ajustó antes de dividir entrenamiento y prueba (fuga menor de datos).
  3. Al haber *frames* consecutivos y 53 duplicados de las mismas sesiones, entrenamiento y prueba comparten muestras muy similares.

---

## 🛡️ Testing, CI/CD & DevSecOps

Para asegurar la integridad y seguridad del software, se implementaron pruebas automatizadas y escaneos de seguridad:
1. **Pruebas Unitarias (PyTest):** Validación de rutas de predicción (`/`, `/predict`), manejo de errores (`422 Unprocessable Entity`) ante vectores malformados y *mocks* autónomos para probar la arquitectura sin cargar el modelo `.keras`.
2. **Pruebas de Carga y Rendimiento:** Latencia promedio estable de **3.14 ms** bajo estrés de **10 Usuarios Virtuales (VUs)** continuos durante 40 segundos con **0 % de tasa de fallo**.
3. **Seguridad (DevSecOps):** Análisis estático (SAST) con **Semgrep** (0 vulnerabilidades bloqueantes) y auditoría de dependencias con `safety` CLI (0 CVEs reportados).

---

## 🌐 Implementación en Entorno Real y Consideraciones Éticas

### Plan de Despliegue por Fases y Costo–Beneficio
1. **Fase Piloto:** Ejecución 100 % local en computador estándar con webcam (OpenCV + MediaPipe + Keras), sin costos de nube ni dependencia de internet.
2. **Fase de Validación:** Pruebas en campo con usuarios externos, distintas condiciones de iluminación y distancias, midiendo latencia real y exactitud.
3. **Fase de Escalamiento:** Incorporación del abecedario completo y modelos secuenciales (LSTM/Transformers) para palabras y gestos dinámicos.
- **Costo–Beneficio:** Los costos se concentran en recolección de datos con consentimiento y mantenimiento; el beneficio principal es una herramienta accesible, de baja latencia y sin exposición de video en servidores externos.

### Consideraciones Éticas
- **Privacidad:** Procesamiento estrictamente local; la aplicación en vivo no almacena ni transmite imágenes del usuario.
- **Sesgos y Equidad:** Necesidad de ampliar el dataset con diversidad de tonos de piel, tamaños de mano y lateralidad (zurdos/diestros), midiendo el desempeño por subgrupo.
- **Transparencia:** Mostrar el porcentaje de confianza de cada predicción e implementar un umbral mínimo para abstenerse ante gestos desconocidos.
- **Impacto Social:** Se concibe como una herramienta de apoyo educativo y comunicativo; **no sustituye** a los intérpretes humanos ni abarca la complejidad gramatical completa de la LSC.

---

## 📂 Estructura del Proyecto

```text
Hands-Interpreter/
├── data/
│   └── dataset/<LETRA>/        # 3.860 imágenes (A, B, C, D, H, L, O)
├── docs/
│   └── img/                    # Gráficas de EDA, entrenamiento y matriz de confusión
├── src/
│   ├── main/
│   │   ├── main.py             # Punto de entrada
│   │   └── mainGestor.py       # Bucle principal de ejecución
│   ├── camera_gestor/
│   │   ├── cameraManager.py    # Gestión de webcam y renderizado
│   │   ├── mediaPipeManager.py # Extracción de landmarks e inferencia
│   │   └── saveDataset.py      # Utilidad de captura de muestras
│   └── Model/
│       ├── neuronalNetwork.py  # CNN 1D, Data Augmentation y métricas
│       └── dataGestor.py       # Exportación de landmarks a CSV
├── hand_model.keras            # Modelo CNN 1D entrenado
├── scaler.pkl                  # Escalador StandardScaler
└── README.md

```
## 🛠️ Instalación y Uso
- **Python:** Version **3.11.2** o superior.  
recomendamos usar un entorno virtual para probar el proyecto
```bash
# Clone the repository
git clone https://github.com/AndresVasped/Hands-Interpreter.git
cd Hands-Interpreter
# Create a virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
# Install dependencies (Requires a requirements.txt file)
pip install -r requirements.txt
```
## ▶️ Execution
Para iniciar el programa, copia este codigo en tu terminal
```bash
python -m src.main.main
```
(recomendamos encarecidamente probar el programa en un lugar donde la luz no afecte mucho el programa)

## 📈 Trabajo Futuro
- [ ] Separar un conjunto de validación independiente, ajustar StandardScaler únicamente con entrenamiento y evaluar con personas externas al dataset.
- [ ] Agregar umbral de confianza mínimo para evitar predicciones forzadas.
- [ ] Limpiar imágenes duplicadas y ampliar el dataset a todo el alfabeto.
- [ ] Implementar reconocimiento de palabras dinámicas e integración con Text-to-Speech (TTS).
