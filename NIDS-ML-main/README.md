# 🛡️ Network Intrusion Detection System using Machine Learning (NIDS-ML)

> **Final Year CSE Project** - An intelligent Network Intrusion Detection System combining Machine Learning, Deep Learning, and Explainable AI for real-time threat detection.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Module Description](#-module-description)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [Future Enhancements](#-future-enhancements)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🎯 Project Overview

This project develops an **intelligent Network Intrusion Detection System (NIDS)** that automatically detects and classifies malicious network traffic in real-time using advanced Machine Learning and Deep Learning techniques.

### Primary Goals

1. **Offline Detection Phase** – Train and evaluate multiple ML models on labeled network traffic data
2. **Real-Time Detection Phase** – Capture live packets using Scapy and classify traffic on the fly
3. **Explainability (XAI)** – Use SHAP to visualize and justify each model decision
4. **Interactive Dashboard** – Build a Streamlit interface to display metrics, live detections, and feature importances

---

## ✨ Features

- 🎯 **Multi-Model Ensemble**: Random Forest, XGBoost, SVM, KNN, Deep Learning (LSTM/CNN)
- 🔍 **Real-Time Detection**: Live packet capture and classification using Scapy
- 🧠 **Explainable AI**: SHAP-based interpretability for model decisions
- 📊 **Interactive Dashboard**: Streamlit-based web interface for monitoring
- ⚖️ **Class Balancing**: SMOTE implementation for handling imbalanced datasets
- 🎨 **Comprehensive Visualization**: Performance metrics, confusion matrices, ROC curves
- 🚨 **Alert System**: Real-time intrusion alerts and logging
- 📈 **Historical Analytics**: Attack pattern analysis and trends

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NETWORK TRAFFIC SOURCE                        │
│                    (Live Packets / Dataset Files)                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PREPROCESSING MODULE                       │
│   • Load Dataset (CICIDS 2017 / NSL-KDD)                           │
│   • Clean Data (Remove duplicates, handle missing values)           │
│   • Encode Features (Label/One-Hot Encoding)                        │
│   • Normalize Features (StandardScaler/MinMaxScaler)                │
│   • Balance Dataset (SMOTE)                                         │
│   • Train-Test Split                                                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE SELECTION MODULE                         │
│   • Correlation Analysis                                            │
│   • Univariate Selection (Chi-square, ANOVA)                       │
│   • Tree-Based Importance (Random Forest)                          │
│   • Recursive Feature Elimination (RFE)                            │
│   • Dimensionality Reduction (PCA - Optional)                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL TRAINING MODULE                          │
│                                                                      │
│   ┌──────────────────────┐    ┌──────────────────────┐            │
│   │  Traditional ML      │    │   Deep Learning      │            │
│   │  • Random Forest     │    │   • LSTM             │            │
│   │  • XGBoost           │    │   • CNN              │            │
│   │  • SVM               │    │   • CNN-LSTM Hybrid  │            │
│   │  • KNN               │    └──────────────────────┘            │
│   │  • Logistic Reg.     │                                        │
│   │  • Naive Bayes       │                                        │
│   └──────────────────────┘                                        │
│                                                                      │
│   • Hyperparameter Tuning (GridSearchCV/RandomizedSearchCV)        │
│   • Cross-Validation                                                │
│   • Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SHAP EXPLAINABILITY MODULE                         │
│   • Generate SHAP Values                                            │
│   • Global Feature Importance                                       │
│   • Local Prediction Explanations                                   │
│   • Summary Plots, Force Plots, Waterfall Plots                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   REAL-TIME DETECTION MODULE                         │
│   • Live Packet Capture (Scapy)                                    │
│   • Feature Extraction from Packets                                 │
│   • Real-Time Classification                                        │
│   • Intrusion Logging & Alerts                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INTERACTIVE DASHBOARD                            │
│                         (Streamlit)                                  │
│   • Live Traffic Monitoring                                         │
│   • Model Performance Metrics                                       │
│   • SHAP Visualizations                                            │
│   • Historical Analytics                                            │
│   • Alert Management                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

### Primary: CICIDS 2017 Dataset

- **Source**: Canadian Institute for Cybersecurity
- **Features**: 80+ network traffic features
- **Attack Types**: DDoS, PortScan, Brute Force, Web Attacks, Infiltration, Botnet
- **Size**: ~2.8M records

### Fallback: NSL-KDD Dataset

- **Source**: NSL-KDD (improved version of KDD Cup 99)
- **Features**: 41 network features
- **Attack Categories**: DoS, Probe, R2L, U2R

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Administrator/Root privileges (for packet capture)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Priyanshu-Ku/NIDS-ML.git
cd "Network Intrusion Detection using ML (NIDS)/NIDS-ML"
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Download CICIDS 2017 from [here](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Extract and place CSV files in `data/raw/` folder
3. Or use NSL-KDD as a fallback

### Step 5: Verify Installation

```bash
python -c "import numpy, pandas, sklearn, tensorflow, scapy; print('All packages installed successfully!')"
```

---

## 💻 Usage

### 1. Data Preprocessing

```bash
cd src
python data_preprocessing.py
```

This will:

- Load the raw dataset
- Clean and preprocess data
- Apply encoding and normalization
- Balance classes using SMOTE
- Save preprocessed data

### 2. Feature Selection

```bash
python feature_selection.py
```

Select the most important features for better model performance.

### 3. Model Training

```bash
python model_training.py
```

This will:

- Train multiple ML models
- Perform hyperparameter tuning
- Evaluate model performance
- Save trained models to `models/` folder

### 4. SHAP Explainability

```bash
python shap_explainability.py
```

Generate SHAP explanations and visualizations.

### 5. Real-Time Detection

**Note: Requires administrator/root privileges**

```bash
# Windows (Run as Administrator)
python realtime_detection.py

# Linux/Mac
sudo python realtime_detection.py
```

### 6. Launch Dashboard

```bash
streamlit run dashboard.py
```

Access the dashboard at `http://localhost:8501`

---

## 📁 Module Description

### `data_preprocessing.py`

Handles all data preprocessing tasks including loading, cleaning, encoding, normalization, balancing, and train-test splitting.

**Key Classes:**

- `DataPreprocessor`: Main preprocessing pipeline

**Key Functions:**

- `load_data()`: Load dataset from CSV
- `clean_data()`: Remove duplicates and handle missing values
- `encode_features()`: Encode categorical features
- `normalize_features()`: Standardize/normalize features
- `balance_dataset()`: Apply SMOTE for class balancing
- `split_data()`: Train-test split with stratification

---

### `feature_selection.py`

Implements various feature selection techniques to identify the most important features.

**Key Classes:**

- `FeatureSelector`: Feature selection pipeline

**Key Methods:**

- `correlation_analysis()`: Remove highly correlated features
- `univariate_selection()`: Chi-square, ANOVA F-test
- `tree_based_importance()`: Random Forest importance
- `recursive_feature_elimination()`: RFE
- `apply_pca()`: Dimensionality reduction

---

### `model_training.py`

Trains and evaluates multiple ML and DL models.

**Key Classes:**

- `MLModelTrainer`: Traditional ML model training
- `DeepLearningTrainer`: Deep learning model training

**Key Models:**

- Random Forest, XGBoost, SVM, KNN, Logistic Regression, Naive Bayes
- LSTM, CNN, CNN-LSTM Hybrid

**Key Features:**

- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Comprehensive evaluation metrics

---

### `shap_explainability.py`

Provides explainability for model predictions using SHAP.

**Key Classes:**

- `SHAPExplainer`: SHAP-based explainability

**Key Visualizations:**

- Summary plots (global importance)
- Force plots (individual predictions)
- Waterfall plots (feature contributions)
- Dependence plots (feature interactions)

---

### `realtime_detection.py`

Captures and analyzes live network traffic in real-time.

**Key Classes:**

- `PacketFeatureExtractor`: Extract features from packets
- `RealtimeDetector`: Real-time classification engine

**Key Features:**

- Live packet capture using Scapy
- Real-time feature extraction
- On-the-fly classification
- Intrusion logging and alerts

---

### `dashboard.py`

Interactive Streamlit dashboard for visualization and monitoring.

**Key Pages:**

- 🏠 Home: Overview and system status
- 📊 Model Performance: Comparison and metrics
- 🔴 Live Detection: Real-time monitoring
- 🧠 Explainability: SHAP visualizations
- 📈 Analytics: Historical data analysis
- ⚙️ Settings: Configuration options

---

## 📈 Results

_(To be updated after model training)_

### Expected Performance Metrics

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| -------------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest        | 98.5%    | 98.3%     | 98.7%  | 98.5%    | 99.2%   |
| XGBoost              | 98.2%    | 98.0%     | 98.4%  | 98.2%    | 99.0%   |
| SVM                  | 96.8%    | 96.5%     | 97.0%  | 96.7%    | 98.5%   |
| Deep Learning (LSTM) | 97.5%    | 97.2%     | 97.8%  | 97.5%    | 98.8%   |

---

## 🛠️ Technologies Used

### Programming Languages

- Python 3.10+

### Machine Learning & AI

- **scikit-learn**: Traditional ML algorithms
- **XGBoost**: Gradient boosting
- **TensorFlow/PyTorch**: Deep learning
- **imbalanced-learn**: SMOTE for class balancing
- **SHAP**: Explainable AI

### Data Processing

- **pandas**: Data manipulation
- **NumPy**: Numerical computing
- **scipy**: Scientific computing

### Network & Security

- **Scapy**: Packet capture and analysis

### Visualization

- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Dashboard framework

### Utilities

- **joblib**: Model persistence
- **logging**: System logging
- **tqdm**: Progress bars

---

## 🔮 Future Enhancements

1. **Ensemble Stacking**: Combine multiple models for better accuracy
2. **Distributed Detection**: Multi-node deployment for large networks
3. **Auto-ML**: Automated model selection and hyperparameter tuning
4. **Advanced DL**: Transformer-based models, GNNs for network graphs
5. **Edge Deployment**: Deploy on edge devices (Raspberry Pi, IoT gateways)
6. **Cloud Integration**: AWS/Azure deployment with auto-scaling
7. **Zero-Day Detection**: Anomaly detection for unknown attacks
8. **Mobile App**: Android/iOS app for mobile monitoring
9. **API Development**: REST API for integration with SIEM systems
10. **Blockchain Logging**: Immutable intrusion logs using blockchain

---

## 👥 Contributors

- **[Your Name]** - _Lead Developer_ - [GitHub Profile]
- **[Teammate 1]** - _ML Engineer_ - [GitHub Profile]
- **[Teammate 2]** - _Data Scientist_ - [GitHub Profile]

**Supervisor:** [Supervisor Name]  
**Institution:** [Your College/University]  
**Year:** 2024-2025

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Canadian Institute for Cybersecurity for CICIDS 2017 dataset
- NSL-KDD dataset providers
- Open-source community for amazing tools and libraries
- Our project supervisor for guidance and support

---

## 📞 Contact

For questions or collaboration:

- Email: [your.email@example.com]
- LinkedIn: [Your LinkedIn]
- Project Link: [GitHub Repository URL]

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{nids-ml-2025,
  author = {Your Name},
  title = {Network Intrusion Detection System using Machine Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/nids-ml}}
}
```

---

**⭐ Star this repository if you find it helpful!**

**🐛 Report bugs and suggest features in the [Issues](https://github.com/yourusername/nids-ml/issues) section.**

---

_Last Updated: November 10, 2025_
