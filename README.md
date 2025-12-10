## End to End Data science project
# 🎓 Student Performance Prediction – End-to-End Machine Learning Project

This project is a complete **end-to-end Machine Learning pipeline** that predicts **students' math scores** based on their academic and personal background.  
It follows a **production-style ML workflow** including data ingestion, preprocessing, model training, evaluation, and experiment tracking using **MLflow & DagsHub**.

---

## 🚀 Features

- ✅ End-to-end ML pipeline (Data → Model → Prediction)
- ✅ Automated data preprocessing using Scikit-learn Pipelines
- ✅ Multiple regression models trained and compared
- ✅ Hyperparameter tuning using GridSearchCV
- ✅ Best model selected using R² score
- ✅ MLflow + DagsHub for experiment tracking
- ✅ Modular, production-ready Python project structure
- ✅ Custom logging and exception handling

---

## 🧠 Problem Statement

To predict a student's **math score** using the following features:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

This is a **regression problem**, and the goal is to achieve the highest possible prediction accuracy.

---

## 🏗️ Project Architecture

mlproject/

│

├── artifacts/ # Saved models & transformed data

├── notebook/ # EDA & training notebooks

│ ├── EDA STUDENT PERFORMANCE.ipynb

│ └── MODEL TRAINING.ipynb

│

├── src/mlproject/

│ ├── components/

│ │ ├── data_ingestion.py

│ │ ├── data_transformation.py

│ │ └── model_trainer.py

│ │

│ ├── exception.py

│ ├── logger.py

│ └── utils.py

│

├── app.py

├── requirements.txt

└── README.md


---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **ML Models:**  
  - Random Forest  
  - Gradient Boosting  
  - XGBoost  
  - CatBoost  
  - AdaBoost  
  - Linear Regression  
- **Experiment Tracking:** MLflow + DagsHub  
- **Version Control:** Git & GitHub  

---

## 📊 Model Performance

- ✅ Best Model Performance: **R² = 0.8802**
- ✅ Hyperparameter tuning done using GridSearchCV
- ✅ Best model saved as: `artifacts/model.pkl`

---

## 🧪 ML Workflow

1️⃣ **Data Ingestion**
- Reads raw dataset and splits it into train & test datasets.

2️⃣ **Data Transformation**
- Handles missing values
- Applies feature scaling
- Encodes categorical features using pipelines

3️⃣ **Model Training**
- Trains 7+ regression models
- Performs hyperparameter tuning
- Selects best model using R² score

4️⃣ **MLflow Tracking**
- Logs:
  - Metrics (RMSE, MAE, R²)
  - Model parameters
  - Model artifacts

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
Install Dependencies
```
pip install -r requirements.txt
```
Run the Application
```
python app.py
```
📈 MLflow Tracking

All experiments are tracked using MLflow integrated with DagsHub.

You can view the experiment dashboard here:

🔗 MLflow Experiment Link:
https://dagshub.com/himitnarayan/data-science-end-to-end.mlflow

  MLFLOW_TRACKING_PASSWORD=e5f59609fba774117e7539818207c3ea4cba1bb2 \
  MLFLOW_TRACKING_URI=https://dagshub.com/himitnarayan/data-science-end-to-end.mlflow \

  MLFLOW_TRACKING_USERNAME=himitnarayan \

python script.py
