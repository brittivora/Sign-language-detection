# Real-Time Sign Language Detection

A real-time sign language recognition system built using **MediaPipe**, **OpenCV**, **TensorFlow**, and **Streamlit**. The system detects and classifies hand gestures (`hello`, `thanks`, `yes`, `no`) using webcam input and deep learning-based sequence modeling.

---

## 📌 Project Overview

This project captures and processes hand landmarks from webcam video frames, builds temporal sequences, and classifies them using an **LSTM neural network**. It includes:

- **Data Collection** using OpenCV
- **Keypoint Extraction** using MediaPipe Hands
- **Sequence Modeling** using LSTM
- **Real-Time Inference** using Streamlit web interface

---

## 🧠 Model Architecture

- **Input**: Sequences of 10 frames, each frame represented as a 126-dimensional vector (2 hands × 21 landmarks × 3D coords).
- **Model**: 
  - LSTM layers: `[64 → 64 → 32]`
  - Dense layers: `[64 → 32 → 4 (softmax)]`
- **Framework**: TensorFlow/Keras
- **Output**: Probability distribution over 4 classes: `hello`, `thanks`, `yes`, `no`.

---


