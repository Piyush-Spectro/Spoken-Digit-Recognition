# Spoken Digit Recognition using ResNet-18 & Mel Spectrograms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Kaggle Score](https://img.shields.io/badge/Kaggle_Test_Accuracy-99.200%25-brightgreen.svg)]()
[![Course](https://img.shields.io/badge/Course-EE708--IIT_Kanpur-orange.svg)]()

An end-to-end deep learning framework for Spoken Digit Recognition (SDR), classifying short audio clips of digits from **0 to 9**. Developed from scratch under strict project constraints without pretrained encoders (e.g., wav2vec or Whisper) or external noise datasets.

---

## 🚀 Key Features & Highlights

- **Kaggle Test Accuracy**: **99.200%** (Validation Score: **~99.17%**)
- **Audio-to-Vision Paradigm**: Converts 1D raw waveforms into 2D **Log-Mel Spectrograms** (64 Mel bands, 16kHz sample rate).
- **Custom ResNet-18**: Modified 1-channel `conv1` input layer and 10-class linear classification head trained from random initialization.
- **SpecAugment Regularization**: Dynamic **Time Masking** ($T=35$) and **Frequency Masking** ($F=15$) applied directly during training to handle unseen speakers and noisy test environments.
- **Optimization Strategy**: Trained with **AdamW** optimizer and **Cosine Annealing** learning rate scheduler over 40 epochs in PyTorch.

---

## 📂 Project Structure

```
Spoken-Digit-Recognition/
├── src/
│   ├── dataset.py        # Custom Audio Dataset loader with MelSpectrogram & SpecAugment
│   ├── model.py          # Modified 1-channel ResNet-18 model architecture
│   └── train.py          # Training loop with validation & model checkpointing
├── notebooks/
│   └── spoken_digit_recognition.ipynb   # Full interactive Jupyter Notebook
├── .gitignore
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
```

---

## 📦 Installation & Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Piyush-Spectro/Spoken-Digit-Recognition.git
   cd Spoken-Digit-Recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   ```bash
   python src/train.py
   ```

---

## 🏆 Model Evaluation

| Metric | Score |
| :--- | :---: |
| **Best Validation Accuracy** | `~99.17%` |
| **Kaggle Blind Test Score** | **`99.200%`** |
| **Total Epochs Trained** | `40` |

---

## 👥 Authors
- **Piyush Kumar** — *Dept. of Mathematics & Scientific Computing, IIT Kanpur*
- Course Project for **EE708: Fundamentals of Data Science & Machine Intelligence**, IIT Kanpur.
