# FarmPulse 🌾🤖
**AI-Verified Agricultural Trade & Smart Farm Security System**

FarmPulse is a comprehensive Agri-Tech platform designed to bridge the gap between farmers and buyers while ensuring farm security. It combines **AI-driven crop quality analysis** with a **real-time intruder detection system**.

## 🚀 Features

### 1. Smart Farm Security (CCTV) 📹
-   **Motion Detection**: Uses pixel-difference algorithms to detect intruders in real-time.
-   **Intruder Alert**: Automatically captures an image of the intruder.
-   **WhatsApp Integration**: Instantly prepares a WhatsApp message to the farm owner with the intruder's image and timestamp.
-   **Siren & TTS**: Triggers a browser-based siren and Text-to-Speech warning ("Security Alert! Motion detected").

### 2. AI Crop Analysis 🌽
-   **Quality Grading**: Analyzes crop images (Corn, Wheat, Tomato, etc.) to determine quality (A/B/C grade).
-   **Disease Diagnosis**: Identifies potential diseases (e.g., Leaf Blight, Rust) and provides treatment recommendations.
-   **Fair Pricing**: Estimates market value based on quality, seasonality, and regional demand.

### 3. Voice-First Interface 🗣️
-   **Multilingual Support**: Full voice guidance in **English, Hindi, Telugu, and Tamil**.
-   **Hands-Free Operation**: Designed for farmers to use easily in the field.

---

## 🛠️ Tech Stack

-   **Frontend**: HTML5, CSS3, Vanilla JavaScript (No frameworks, lightweight).
-   **Backend**: Python (FastAPI).
-   **AI/ML**: TensorFlow/Keras (MobileNetV2), OpenCV, Scikit-learn.
-   **Deployment**: Netlify (Frontend), Render (Backend).

---

## 📦 Installation & Setup

### Prerequisites
-   Python 3.9+
-   Git

### 1. Clone the Repository
```bash
git clone https://github.com/RamatenkiCharan/FARMPULSE1.git
cd FARMPULSE1
```

### 2. Setup Backend (Python)
Navigate to the backend folder and install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

Run the server:
```bash
python server.py
# Server runs at http://localhost:8000
```

### 3. Setup Frontend
Simply open `index.html` in your browser.
*Note: For camera features to work, you must serve it via HTTPS or localhost.*

You can use Python to serve it locally:
```bash
# In the root folder
python -m http.server 8080
# Open http://localhost:8080
```

---

## 🌍 Deployment

### Backend (Render)
1.  Push code to GitHub.
2.  Create a new **Web Service** on Render.
3.  Connect your repo.
4.  Render will auto-detect `render.yaml` or use `pip install -r requirements.txt`.

### Frontend (Netlify)
1.  Drag and drop the project folder to **Netlify Drop**.
2.  **Configuration**:
    -   Open `index.html`.
    -   Update `const API_BASE_URL` to your live Render Backend URL.

---

## 🛡️ License
This project is open-source and available for educational and agricultural development purposes.

---
*Built with ❤️ for Indian Farmers.*
