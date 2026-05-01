# Autoencoders-for-Dimensionality-Reduction
# 🧠 Autoencoders for MNIST (Dimensionality Reduction & Denoising)

## 📌 Project Overview

This project implements different types of autoencoders using PyTorch to perform image compression, reconstruction, and denoising on the MNIST handwritten digits dataset.

The goal is to learn efficient latent representations and analyze reconstruction quality.

---

## 🎯 Objectives

* Implement Vanilla Autoencoder (Fully Connected)
* Implement Convolutional Autoencoder (CNN-based)
* Implement Denoising Autoencoder
* Visualize reconstruction results
* Analyze model performance

---

## ⚙️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-learn

---

## 📂 Project Structure

```id="g5z40g"
Autoencoder-Assignment/
│
├── data/
│
├── src/
│   ├── data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── visualize.py
│
├── main.py
├── requirements.txt
├── README.md
├── report.pdf
```

---

## 📦 Dataset

* Dataset: MNIST Handwritten Digits
* Source: PyTorch (`torchvision.datasets.MNIST`)
* Size: 70,000 images (28×28 grayscale)

---

## 🔁 Workflow

1. Load dataset
2. Normalize pixel values
3. Flatten images (for vanilla autoencoder)
4. Add noise (for denoising autoencoder)
5. Train models
6. Visualize reconstruction results

---

## 🧠 Models Implemented

### 🔹 Vanilla Autoencoder

* Fully connected neural network
* Latent dimension: 32
* Basic compression and reconstruction
* <img width="1083" height="196" alt="Screenshot 2026-05-01 113426" src="https://github.com/user-attachments/assets/55ffaa14-0077-45c4-a1c3-9061f7a033d5" />

### 🔹 Convolutional Autoencoder

* Uses convolutional layers
* Better spatial feature learning
* Improved reconstruction quality
<img width="1112" height="221" alt="Screenshot 2026-05-01 113440" src="https://github.com/user-attachments/assets/ca623c43-ead8-4894-af2b-d0b08174e673" />
  

### 🔹 Denoising Autoencoder

* Trained on noisy images
* Reconstructs clean images
* Handles Gaussian noise
* <img width="1127" height="459" alt="Screenshot 2026-05-01 114742" src="https://github.com/user-attachments/assets/62fc5d67-bd5e-4b6c-9c59-47b50b4731f9" /> 

---

## 📊 Results

* Vanilla AE produces basic reconstructions
* Conv AE gives sharper outputs
* Denoising AE removes noise effectively

---

## 🚀 How to Run

### 1. Install Dependencies

```id="3j1t4n"
pip install -r requirements.txt
```

### 2. Run Project

```id="q3v4i1"
python main.py
```

---

## 📌 Conclusion

Autoencoders are powerful tools for unsupervised learning. Convolutional autoencoders outperform vanilla models for image data, while denoising autoencoders effectively restore corrupted inputs.


Student Assignment Submission

