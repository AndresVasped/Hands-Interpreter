# Hands Interpreter
nuestro programa se basa en un lector de lenguaje de señas, esto permite:

- Reconocer en tiempo real las letras que el usuario realiza con sus manos.  
- Mostrar los resultados de forma intuitiva en pantalla.  
- Servir como base para futuras aplicaciones educativas e inclusivas.  

---

## 🧩 Patrones de Diseño Utilizados

El diseño del sistema se apoya en **patrones de diseño de software**, lo que garantiza **mantenibilidad, extensibilidad y claridad** del código. Entre ellos:

1. **Patrón Strategy**  
   - Se emplea para manejar distintas formas de reconocimiento (ej. letras actuales, en un futuro números o palabras).  
   - Permite cambiar el algoritmo de reconocimiento sin modificar el resto del sistema.  

2. **Patrón Singleton**  
   - Se utiliza en la clase que maneja la cámara, asegurando que solo exista **una única instancia activa** controlando el video.  

3. **Patrón Factory**  
   - Permirte la creacion de distintos objetos que interactuan entre si.  

---

## 🚀 Usos y Aplicaciones

- **Educación inclusiva:** apoyo a personas que deseen aprender el lenguaje de señas.  
- **Comunicación asistida:** servir como base para sistemas que traduzcan señas a texto o voz.  
- **Investigación:** aplicación en proyectos de accesibilidad y visión artificial.  
- **Práctica personal:** herramienta para estudiantes de lengua de señas que quieran validar sus gestos.  

---

## ⚙️ Tecnologías Utilizadas

- **Python** como lenguaje principal.  
- **OpenCV** para el manejo de la cámara y procesamiento de imágenes.  
- **MediaPipe** para la detección de manos y obtención de puntos clave.  
- **Numpy / Matplotlib** (opcional) para cálculos y visualizaciones.  

---

## 📈 Futuras Mejoras

- Reconocimiento de palabras y frases completas.  
- Integración con un sistema de **síntesis de voz**.  
- Creación de una aplicación móvil multiplataforma.  
- Base de datos personalizada para ampliar el vocabulario reconocido.  

---
##  Requisitos del Sistema

- **Python:** versión **3.11.2** o superior.  
- **Librerías necesarias:**  
  - OpenCV  
  - MediaPipe  
  - TensorFlow  
  - Keras  
  - NumPy  
  - Pandas  

---

## ▶️ Ejecución del Programa

Para iniciar el programa, usar el siguiente comando en la terminal:

```bash
python -m src.main.main
