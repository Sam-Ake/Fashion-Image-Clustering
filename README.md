# End-to-End Fashion Image Clustering & Explorer

An end-to-end computer vision and unsupervised machine learning project that automatically clusters thousands of diverse fashion apparel, footwear, and accessories without relying on manual training labels.

## 🚀 Project Overview
This project builds a scalable image feature extraction and clustering pipeline using deep learning and unsupervised learning techniques:
- **Feature Extraction:** Utilizes a pretrained **ResNet50** convolutional neural network (PyTorch) to convert raw fashion imagery into dense **2048-dimensional feature vectors**.
- **Dimensionality Reduction:** Applies **Principal Component Analysis (PCA)** to reduce feature space and mitigate the curse of dimensionality.
- **Unsupervised Clustering:** Groups similar items using **KMeans Clustering**.
- **Interactive Web App:** Deployed via **Streamlit** to explore cluster groupings visually.

---

## 🛠️ Tech Stack & Libraries
- **Core:** Python, Jupyter Notebooks
- **Deep Learning & Vision:** PyTorch, torchvision, PIL (Pillow)
- **Data & Machine Learning:** pandas, NumPy, scikit-learn (PCA, KMeans), joblib
- **Visualization:** matplotlib, seaborn, Streamlit

---

## 📁 Project Structure
```text
fashion-dataset/
│
├── .gitignore
├── requirements.txt
├── app.py                      # Streamlit Web Application
├── pca_model.pkl               # Saved PCA transformer artifact
├── kmeans_model.pkl            # Saved KMeans clustering model artifact
├── clustered_fashion_sample.csv # Processed metadata with cluster labels
├── notebooks/                  # EDA & feature extraction notebooks
└── FASHION-DATASET/            # Image directory and source CSVs (local)