# End_to_End_Multi_Omics_Integration_Pipeline_Multi_View_Autoencoders_for_Disease
End_to_End_Multi_Omics_Integration_Pipeline_Multi_View_Autoencoders_for_Disease_Subtyping_and_Classification
# End-to-End Multi-Omics Integration Pipeline: Multi-View Autoencoders for Disease Subtyping

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Overview

This repository implements an **End-to-End Multi-Omics Integration Pipeline** designed to classify disease subtypes and identify biological patterns by combining heterogeneous omics data sources.

Traditional methods often struggle to capture non-linear interactions between different omics layers (e.g., Genomics, Transcriptomics, Proteomics). This project utilizes **Multi-View Autoencoders** (Deep Learning) to learn a compressed, shared latent representation of the data, which significantly improves clustering and downstream classification performance compared to standard dimensionality reduction techniques like PCA.

## 🚀 Key Features

* **Synthetic Data Generation**: Simulates high-dimensional multi-omics datasets (RNA-seq, Proteomics, miRNA-seq/Exome) with distinct disease signals.
* **Data Preprocessing**: Implementation of standard normalization (Log1p) and scaling (StandardScaler) pipelines.
* **Baseline Integration**: Comparison using standard Concatenation + PCA + UMAP.
* **Canonical Correlation Analysis (CCA)**: visualization of linear correlations between RNA and Proteomics views.
* **Deep Learning Models**:
    * **Multi-Input Neural Network**: A supervised branch-network for direct classification.
    * **Multi-View Autoencoder**: An unsupervised/self-supervised model that learns a shared latent space by compressing and reconstructing multiple omics views simultaneously.
* **Latent Space Visualization**: UMAP projections of autoencoder embeddings showing clear separation of disease vs. healthy samples.
* **Downstream Classification**: Benchmarking Random Forest and MLP classifiers on the learned latent embeddings (achieving ~0.98 - 1.0 ROC AUC).

## 🛠️ Technologies & Libraries

The pipeline is built using the following technologies:

* **Python 3.x**
* **TensorFlow / Keras**: For building the Multi-View Autoencoder and Neural Networks.
* **Scikit-Learn**: For PCA, CCA, StandardScaler, Random Forest, and evaluation metrics.
* **UMAP-Learn**: For non-linear dimensionality reduction and visualization.
* **Pandas & NumPy**: For data manipulation and linear algebra operations.
* **Matplotlib & Seaborn**: For plotting and data visualization.

## 📂 Dataset

**Note:** This project generates **synthetic data** internally to ensure reproducibility and privacy. No external data download is required to run the notebook.

The generator creates:
1.  **RNA-seq**: Continuous gene expression data (Gamma distribution).
2.  **Proteomics**: Protein abundance data (Normal distribution).
3.  **Exome/miRNA**: Variant or expression data (Poisson/Binary distributions).
4.  **Labels**: Synthetic Binary Labels (Healthy vs. Disease).

## ⚙️ Pipeline Architecture

### 1. Preprocessing
Data is log-transformed to handle skewness typical in omics data and standardized (Z-score normalization) to ensure all features contribute equally to the loss function.

### 2. Multi-View Autoencoder
The core of the project is a Neural Network with the following structure:
* **Encoders**: Separate dense layers for each omics view (Exome, RNA, Proteomics) to extract view-specific features.
* **Fusion Layer**: Concatenates features from all views.
* **Latent Bottleneck**: A compressed layer (dim=64) that forces the network to learn the most essential shared biological signals.
* **Decoders**: Separate branches attempting to reconstruct the original input data from the shared latent representation.

### 3. Downstream Analysis
The **Latent Embeddings** are extracted from the trained autoencoder and used as input for:
* **UMAP Visualization**: To visually inspect clustering.
* **Supervised Classification**: Random Forest and Dense Neural Networks are trained on these embeddings to predict disease labels.

## 📊 Results

The pipeline demonstrates that integrating data via Autoencoders yields superior feature representations:

| Model / Method | Metric | Score | Description |
| :--- | :--- | :--- | :--- |
| **Random Forest (on Latent)** | ROC AUC | **1.00** | Perfect separation on synthetic test data |
| **Cross-Validation (RF)** | ROC AUC | **0.98** | Robust performance across folds |
| **Neural Classifier** | Accuracy | **High** | Converges to near-perfect accuracy |

*Visualizations in the notebook show distinct clustering of disease groups in the Autoencoder latent space compared to raw data.*

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/multi-omics-integration-pipeline.git](https://github.com/your-username/multi-omics-integration-pipeline.git)
    cd multi-omics-integration-pipeline
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn tensorflow umap-learn mofapy2
    ```

4.  **Run the Notebook:**
    ```bash
    jupyter notebook End_to_End_Multi_Omics_Integration_Pipeline.ipynb
    ```

## 📜 File Structure
