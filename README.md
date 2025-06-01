# Alzheimer's Disease Detection using CNN

This project focuses on the automatic detection of Alzheimer's Disease using Convolutional Neural Networks (CNN). It aims to assist in early diagnosis by classifying brain MRI images into different stages of Alzheimer's, leveraging deep learning techniques.

## 🧠 Project Overview

Alzheimer’s Disease is a progressive neurological disorder that affects memory and cognitive abilities. MRI scans of the brain can help identify early signs of the disease. This project uses CNN to classify MRI images into the following classes:
- **Non-Demented**
- **Very Mild Demented**
- **Mild Demented**
- **Moderate Demented**

The goal is to develop a model that can help in early and accurate detection, improving chances of effective management.

## 📁 Dataset

The dataset used in this project is the **Alzheimer’s Dataset (4 class of Images)**, originally from Kaggle.

- 4 Classes: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`
- Source: [Kaggle - Alzheimer MRI Dataset](https://www.kaggle.com/datasets/bindumadhavi25/ad-dataset)

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## 🧪 Project Structure
Alzheimers-Detection-using-CNN/
│
├── data/ # Dataset (not included due to size)
├── models/ # Saved model weights
├── images/ # Visual results and graphs
├── cnn_model.py # CNN architecture and training
├── utils.py # Helper functions for preprocessing and evaluation
├── Alzheimer_CNN.ipynb # Jupyter Notebook with complete pipeline
└── README.md # Project documentation


## 📌 Steps Performed

1. **Import Libraries**
2. **Load and Explore Dataset**
3. **Preprocessing and Data Augmentation**
4. **Train-Test Split**
5. **CNN Model Design**
6. **Model Training and Validation**
7. **Evaluation using Accuracy, Confusion Matrix**
8. **Visualization of Results**
9. **Saving Model**

## 🔍 Key Results

- Achieved accuracy of **99%** on validation set .
- Model performs well in distinguishing between different stages of Alzheimer’s.

## 📈 Sample Visualizations

- Training vs. Validation Accuracy & Loss
- Confusion Matrix
- Prediction Samples with Labels


## 🚀 Future Work

- Incorporate **Vision Transformers (ViTs)** to improve accuracy.
- Develop a **hybrid model combining CNN and ViT**.
- Deploy as a web application using Streamlit or Flask.



