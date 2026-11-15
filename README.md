# 🤟 Hands Interpreter: Real-Time Sign Language Recognition
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
Our program is a real-time sign language interpreter built with Python and Computer Vision. This project allows users to:
* **Recognize real-time hand gestures** representing letters of the alphabet.
* **Display results intuitively** on the screen overlay.
* **Serve as an educational baseline** for inclusive communication applications.
---
## 🏗️ Software Architecture & Design Patterns
Since scalability and maintainability were core goals for this project, the system design heavily relies on **Software Design Patterns**. This ensures clean code and easy feature expansion:
1. **Strategy Pattern:** Used to manage different recognition algorithms (e.g., current letter recognition, future numbers, or dynamic gestures). Allows swapping recognition behaviors at runtime without modifying the core inference engine.
2. **Singleton Pattern:** Applied to the Camera/Video Manager class. Ensures that only **one active instance** controls the hardware video stream, preventing resource blocking and memory leaks.
3. **Factory Pattern:** Centralizes the instantiation of various interconnected objects (UI overlays, recognition modules), making the system highly modular.
---
## 🧠 How it Works (The ML Pipeline)
1. **Capture:** OpenCV captures the real-time video feed from the webcam.
2. **Feature Extraction:** Google's **MediaPipe Hands** processes the frame to extract 21 3D hand landmarks (keypoints).
3. **Inference:** The coordinates are normalized and passed to a pre-trained **Dense Neural Network (TensorFlow/Keras)** (`hand_model.keras`).
4. **Output:** The model classifies the coordinate array into a specific letter, and OpenCV renders the bounding box and text on the screen.
### 📸 Custom Dataset Generation
One of the most valuable aspects of this project is that we did not rely on pre-existing image datasets from the internet. We developed a custom data collection module that allows the system to learn directly from the user:
* **Dynamic Filtering:** Instead of saving raw, heavy images (which can cause overfitting due to background noise), it uses MediaPipe to verify if a hand is clearly visible in the frame.
* **Coordinate Extraction:** If a hand is detected, it extracts the 63 spatial points (21 landmarks × X, Y, Z coordinates) and saves exactly those numeric values into our dataset, assigning them the corresponding label (e.g., Letter 'A').
This approach made the training process incredibly fast, lightweight, and scalable.
---
## 🛡️ Testing, CI/CD & DevSecOps
To ensure the integrity, scalability, and security of the application, we implemented a robust testing and security phase, atypical for standard academic projects:
### 1. Software Testing (PyTest)
We created a suite of automated tests to validate the endpoints and the AI model:
* **Unit Tests:** Verified API route health (`/`, `/predict`) and ensured correct error handling (`422 Unprocessable Entity`) for malformed keypoint payloads.
* **Mock Autonomous Tests:** Simulated the prediction flow without loading the heavy `.keras` model into memory, validating the software architecture independently of the ML layer.
### 2. Load Testing & Performance
We tested the concurrent capabilities of the AI prediction endpoint:
* **Results:** Maintained a stable average latency of **3.14ms**.
* **Stress:** Managed 10 Virtual Users (VUs) constantly bombarding the API for 40 seconds with **0% failure rate**.
### 3. Vulnerability Scanning (DevSecOps)
* **Code Security:** We used **Semgrep** to perform static application security testing (SAST), resulting in 0 blocking findings or code vulnerabilities.
* **Dependency Safety:** We used the `safety` CLI tool to scan our `requirements.txt` against open-source vulnerability databases. The environment was certified stable with 0 CVEs reported.
### 4. Neural Network Validation
During the training phase of the Keras Sequential Model (63 Inputs -> 128 ReLU -> Dropout -> 64 ReLU -> Softmax):
* **Early Stopping** was implemented with a patience level of 15 epochs to retrieve the best weights.
* Both Training Accuracy and Validation Accuracy converged smoothly near 1.00 (100%) without exhibiting significant overfitting, proving the model generalized the coordinates correctly instead of merely memorizing images.
---
## 🚀 Use Cases & Applications
- **Inclusive Education:** Support tool for individuals learning sign language.  
- **Assisted Communication:** Baseline system for translating signs to text or synthesized speech.  
- **Computer Vision Research:** Practical application in accessibility and human-computer interaction (HCI).  
---
## ⚙️ Tech Stack
- **Python** (Core Logic)  
- **OpenCV** (Camera handling and image processing)  
- **MediaPipe** (Hand tracking and landmark extraction)  
- **TensorFlow / Keras** (Deep learning model inference)
- **NumPy & Pandas** (Data manipulation and matrix calculations)  
---
## 🛠️ System Requirements & Setup
- **Python:** Version **3.11.2** or higher.  
It is highly recommended to use a virtual environment. Clone the repository and install the required dependencies:
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
To iniciate the program, copy this code to the root terminal
```bash
python -m src.main.main
```
## 📈 Roadmap & Future Improvements
- [ ] Implementation of full word and phrase recognition.
- [ ] Integration with a **Text-to-Speech (TTS)** synthesizer.
- [ ] Cross-platform mobile application port.
- [ ] UI/UX improvements for the custom dataset generation tool.
