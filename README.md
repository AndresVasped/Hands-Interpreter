# Hands Interpreter
nuestro programa se basa es un lector de lenguaje de señar, esto permite:

- Reconocer en tiempo real las letras que el usuario realiza con sus manos.  
- Mostrar los resultados de forma intuitiva en pantalla.  
- Servir como base para futuras aplicaciones educativas e inclusivas.  

---

## 🧩 Patrones de Diseño Utilizados

El diseño del sistema se apoya en **patrones de diseño de software**, lo que garantiza **mantenibilidad, extensibilidad y claridad** del código. Entre ellos:

1. **Patrón MVC (Modelo - Vista - Controlador)**  
   - **Modelo:** gestiona los datos de las coordenadas de los dedos y el reconocimiento de letras.  
   - **Vista:** muestra la cámara en tiempo real y los resultados de las detecciones.  
   - **Controlador:** conecta la lógica de detección con la interfaz gráfica.  

2. **Patrón Strategy**  
   - Se emplea para manejar distintas formas de reconocimiento (ej. letras actuales, en un futuro números o palabras).  
   - Permite cambiar el algoritmo de reconocimiento sin modificar el resto del sistema.  

3. **Patrón Singleton**  
   - Se utiliza en la clase que maneja la cámara, asegurando que solo exista **una única instancia activa** controlando el video.  

4. **Patrón Observer**  
   - La vista se suscribe a los cambios del reconocimiento, actualizándose automáticamente cuando se detecta una nueva letra.  

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

## 📷 Ejemplo de Uso

1. Ejecutar el programa.  
2. Colocar las manos frente a la cámara.  
3. El sistema detectará la posición de los dedos y mostrará en pantalla la letra correspondiente en tiempo real.  

---
