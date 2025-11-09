# 💳 Credit Card Fraud Detection System

A complete **end-to-end machine learning project** that detects fraudulent credit card transactions using **XGBoost** and a **Streamlit**-based web app.  
This project demonstrates data preprocessing, feature engineering, model training, hyperparameter tuning, and deployment — all using **Object-Oriented Programming (OOP)** principles.

---

## 🚀 Features

✅ End-to-end ML pipeline (data → model → deployment)  
✅ Handles class imbalance using SMOTE  
✅ Feature engineering (`age`, `hour`, `day`, `weekday`, `distance`)  
✅ Model tuning using RandomizedSearchCV  
✅ Optimized F1-score and ROC-AUC metrics  
✅ Fraud detection web app built with Streamlit  
✅ Dynamic UI feedback (red = fraud found, green = safe)  
✅ Downloadable results and automatic background reset  

---

## 🏗️ Project Structure
```
Credit Card Fraud Detection System/
│
├── app/
│ ├── predict_app.py # Streamlit frontend with animation & interactivity
│ └── prediction_pipeline.py # OOP pipeline for preprocessing & prediction
│
├── data/
│ └── creditcard.csv # Raw dataset (replace with your own)
│
├── notebooks/
│ └── eda_and_model_training.ipynb # Jupyter notebook for EDA & model training
│
├── models/
│ └── fraud_model.pkl # Trained XGBoost model (saved with joblib)
│
├── utils/
│ └── preprocessing.py # (Optional) helper functions if needed
│
├── requirements.txt
└── README.md
```


---

## ⚙️ Tech Stack

| Category | Technology |
|-----------|-------------|
| Programming | Python 3.10+ |
| Libraries | pandas, numpy, scikit-learn, xgboost |
| Sampling | imbalanced-learn (SMOTE) |
| Visualization | matplotlib, seaborn |
| Web App | Streamlit |
| Model Persistence | joblib |

---

## 🧠 Workflow Overview

1. **Exploratory Data Analysis (EDA)**
   - Checked for imbalance and null values  
   - Correlation analysis & visualization  

2. **Feature Engineering**
   - Created time-based features: `hour`, `day`, `weekday`  
   - Derived user features: `age`, `distance`  

3. **Model Training**
   - Used **XGBoost** with class-weighting and SMOTE  
   - Tuned parameters using **RandomizedSearchCV**  
   - Optimized **threshold** for best F1-score  

4. **Model Evaluation**
   - Achieved F1-score ≈ `0.79` and ROC-AUC ≈ `0.98`  

5. **Deployment**
   - Created **Streamlit** UI for predictions  
   - Real-time CSV upload, fraud detection.

---

## 🎨 Streamlit App Highlights

| Action | Effect |
|--------|---------|
| Upload CSV | Data preview shown |
| Predict | Runs pipeline and shows frauds |
| Download | Download csv file with predictions |

---

## 🧩 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/dhruv70441/credit-card-fraud-detection-system.git
cd credit-card-fraud-detection-system
```


### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate    # For Windows
# source venv/bin/activate   # For Linux/Mac
```


### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```


### 4️⃣ Train the Model (optional)
```bash
jupyter notebook notebooks/eda_and_model_training.ipynb
```
    -This will generate and save fraud_model.pkl inside the models/ directory.


### 5️⃣ Run the Streamlit App
```bash
streamlit run app/predict_app.py
```


### 6️⃣ Upload Your File

Upload a fraudTest.csv file containing transaction data from Data folder.
The app will preprocess, predict, and highlight fraud transactions dynamically.


👨‍💻 Author

Dhruv Parmar
✉️ dhruvparmar70441@gmail.com

📍 India

If you find this helpful, feel free to ⭐ star this repo on GitHub!

