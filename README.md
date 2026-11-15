# 🤟 Hands Interpreter: Real-Time Sign Language Recognition

![Python](https://shields.io)
![OpenCV](https://shields.io)
![MediaPipe](https://shields.io)
![TensorFlow](https://shields.io)
![PyTest](https://shields.io)
![Semgrep](https://shields.io)

**HOLA**
https://github.com

**ABCD**
https://github.com

Intérprete de lenguaje de señas en tiempo real desarrollado en Python mediante Visión por Computador. El sistema utiliza **MediaPipe** para extraer 21 puntos clave de la mano y un modelo secuencial de **TensorFlow/Keras** (`hand_model.keras`) para clasificar gestos del alfabeto con una arquitectura basada en patrones de diseño (Estrategia, Singleton y Fábrica).

## 🏗 Arquitectura y Pipeline de ML
1. **Captura y Extracción:** OpenCV procesa el video de la cámara web y MediaPipe extrae 63 coordenadas espaciales (X, Y, Z).
2. **Inferencia:** Las coordenadas normalizadas pasan por una red neuronal densa entrenada con un dataset personalizado.
3. **Pruebas y Seguridad:** Incluye validación mediante PyTest, pruebas de carga estables (latencia promedio de 3.14ms) y análisis estático con Semgrep.

## 🚀 Uso y Requisitos

- **Requisitos:** Python 3.11.2+ y un entorno virtual recomendado.
- **Instalación:**
  ```bash
  git clone https://github.com
  cd Hands-Interpreter
  python -m venv venv
  source venv/Scripts/activate  # En Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```
- **Ejecución:**
  ```bash
  python -m src.main.main
  ```

