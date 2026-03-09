# 🌙 LumaNet – Real-Time Low-Light Image Enhancement

🔗 **Live Demo:** https://curr-lumanet.onrender.com/

LumaNet is a real-time low-light image enhancement system built using a Zero-Reference Deep Learning approach (Zero-DCE++).  
It enhances brightness, contrast, and color consistency in dark images without requiring paired training data.

Designed for CPU-level real-time performance and lightweight deployment.

---

## 📌 Project Overview

Low-light conditions significantly degrade computer vision performance (face detection, surveillance, object detection).

Traditional enhancement methods:
- Require paired datasets
- Are computationally heavy
- Introduce artifacts or color distortions

LumaNet solves this using:

✔ Zero-Reference Learning  
✔ Tiny neural network (~10K parameters)  
✔ Real-time CPU inference  
✔ No paired dataset requirement  

---

## 🧠 Core Model: Zero-DCE++

LumaNet is based on:

**Zero-Reference Deep Curve Estimation (Zero-DCE++)**

Instead of learning from ground-truth images, the model estimates a pixel-wise light enhancement curve:

\[
LE(I(x); α) = I(x) + α I(x)(1 - I(x))
\]

The curve is applied iteratively (8 times) for smooth enhancement.

---

## 🏗 Architecture

### 🔹 Input
Low-light RGB image (H × W × 3)

### 🔹 Backbone
7-layer lightweight CNN with:
- Symmetrical skip connections
- Depthwise separable convolutions
- ~10K parameters

### 🔹 Loss Functions (Non-Reference)

- **Spatial Consistency Loss (Lspa)** – Preserves textures
- **Exposure Control Loss (Lexp)** – Controls brightness (~0.6 target)
- **Color Constancy Loss (Lcol)** – Prevents color distortion
- **Illumination Smoothness Loss (LtvA)** – Ensures smooth light transitions

### 🔹 Output
α-parameter map for curve-based brightness correction.

---

## ⚡ Performance

| Metric | LumaNet |
|--------|----------|
| Parameters | ~10K |
| Model Size | ~10 KB |
| CPU Speed | ~11 FPS |
| GPU Speed | ~1000 FPS |
| Learning Type | Zero-Reference |
| Dataset Requirement | No Paired Data |

---

## 🌐 Web Application

The model is deployed using:

- Flask
- PyTorch
- OpenCV
- Gunicorn
- Render.com (Free tier)

Users can:
- Upload a low-light image
- Get enhanced output instantly

---

## 📂 Project Structure

```

curr-lumanet/
│
├── app.py                # Flask server
├── model.pth             # Trained LumaNet model
├── requirements.txt      # Dependencies
├── runtime.txt           # Python version (3.10)
├── Procfile              # Gunicorn start command
├── templates/            # HTML files
├── static/               # CSS / JS
└── README.md

````

---

## 🚀 Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/SuhainaFathimaM/curr-lumanet.git
cd curr-lumanet
````

### 2️⃣ Create virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🧪 API Usage

### POST /predict

Upload an image for enhancement.

Example:

```bash
curl -X POST -F "file=@input.jpg" \
https://curr-lumanet.onrender.com/predict
```

---

## 🛠 Deployment (Render)

Key configuration:

* Python 3.10.13
* `gunicorn app:app`
* Clean requirements file
* CPU-only PyTorch

---

## 📚 Research References

1. Guo et al., Zero-Reference Deep Curve Estimation, CVPR 2020
2. Li et al., Zero-DCE++, TPAMI 2021
3. Ma et al., SCI Framework, CVPR 2022
4. Gao et al., BézierCE, Sensors 2023
5. Yu et al., Zero-TCE, Applied Sciences 2025

---

## 🎯 Objectives

* Real-time enhancement on CPU
* Zero-reference training
* Lightweight architecture (~10K parameters)
* Improve downstream vision tasks (e.g., face detection in dark)

---

## 👩‍💻 Developer

**Suhaina Fathima M**
Domain: Deep Learning & Computer Vision

---

## 📜 License

For academic and research purposes.

---

## 🌟 Future Improvements

* ONNX conversion for faster inference
* Mobile deployment
* Edge-device optimization
* Integration with CCTV / webcam feed

---

✨ Thank you for visiting LumaNet!


