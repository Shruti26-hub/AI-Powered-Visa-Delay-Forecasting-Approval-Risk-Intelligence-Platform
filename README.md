# AI-Powered-Visa-Delay-Forecasting-Approval-Risk-Intelligence-Platform
The AI-Powered Visa Delay Forecasting & Approval Risk Intelligence Platform is a cutting-edge solution designed to demystify the complex world of global immigration. By leveraging advanced machine learning algorithms, this platform provides applicants, legal firms, and HR departments with high-accuracy predictions for visa processing times and a comprehensive risk assessment of application approval. 
It analyzes historical trends, current regulatory environments, and individual application data to identify potential red flags before submission, significantly reducing the uncertainty of international mobility and improving success rates through proactive, data-driven intelligence.

## 🚀 Live Application

You can access the deployed Streamlit dashboard here:

🔗 **Streamlit App:**  
https://mv2dybf9nazkdfqcas2kba.streamlit.app/

This interactive dashboard allows users to:

- Predict visa processing timelines
- Analyze historical visa processing trends
- Evaluate delay risk using machine learning
- Explore AI explainability through SHAP visualizations
- Simulate application scenarios using the interactive simulator



---

##  Project Overview

Project Overview

The AI-Powered Visa Delay Forecasting & Approval Risk Intelligence Platform is a data-driven analytics system designed to reduce uncertainty in employer-sponsored immigration processes.

Using historical U.S. PERM labor certification data, the platform:

Predicts visa processing timelines
Assesses approval risk probability
Identifies workload and backlog patterns
Extracts seasonal and economic signals
Provides structured features for machine learning models

---
Background

PERM (Program Electronic Review Management) is a labor certification process managed by the U.S. Department of Labor. Before sponsoring a foreign worker for permanent employment, employers must receive labor certification approval.

Processing timelines are often long and highly variable due to:
Audit reviews
Backlogs
Regulatory shifts
Wage determination cycles
Industry-specific demand

This project analyzes those patterns to build predictive intelligence.
---
## 📂Dataset Used

The dataset is based on **US PERM Visa Applications** and contains information such as:

- Application submission date  
- Decision date  
- Applicant citizenship  
- Employer details  
- Job and wage information  
- Case outcome (Certified / Denied)

The cleaned dataset was prepared using both:

- **Power BI (initial cleaning)**
- **Python (final preprocessing + feature creation)**

---

##  Cleaning & Preprocessing Steps

---
## Dataset Cleaning & Preprocessing (Module 1)
# 1️⃣ Data Cleaning in Power BI

The raw dataset contained:

- Blank spaces
- Duplicate records
- Incorrect formats
- Missing values

Using Power BI, the following cleaning was performed:

✔ Removed blank/empty values  
✔ Removed duplicate rows  
✔ Corrected column formats (dates, numeric fields)  
✔ Removed errors and invalid entries  
✔ Standardized categorical values  

After Power BI cleaning, the dataset was saved as:

📌 `cleaned visa dataset.xlsx`

---

# 2️⃣ Data Preprocessing in Python

Python was used for final preprocessing and feature engineering.

---
# **Module 2 — Exploratory Data Analysis (EDA)**

EDA was conducted using:

Pandas

Seaborn

Matplotlib

🔹 1. Processing Time Distribution

Right-skewed distribution

Long-tail behavior due to backlog cases

Mean processing time ≈ 3 years

Extreme durations reflect real operational delays

🔹 2. Processing Time by Visa Type

Violin plots showed:

Significant variation across visa categories

Certain visa types exhibit higher median delays

Distribution width indicates variability

🔹 3. Workload vs Processing Time

Aggregated state-level analysis revealed:

Higher application volume correlates with increased delays

Backlog pressure influences processing duration

🔹 4. Seasonal Trend Analysis

Extracted:

Filing year

Filing month

Filing quarter

Findings:

Monthly variation suggests seasonal workload cycles

Yearly trend shows multi-year backlog fluctuations

🔹 5. Wage & Industry Influence

Wage levels show partial correlation with delay

Certain industries experience systematically higher processing times

🧠 Feature Engineering

Based on EDA insights, the following features were engineered:

📅 Time-Based Features

processing_days

filing_year

filing_month

filing_quarter

decision_year

🏢 Employer Features

employer_age

📊 Workload Feature

monthly_volume (application count per month)

🌍 Demand Pressure Features

country_frequency

visa_frequency



These features capture:
Seasonal trends
Structural backlog
Economic signals
Employer maturity
Demand pressure

📈 Extreme Value Treatment

Instead of removing rare long-duration cases, the project used:
99th percentile capping (Winsorization)

Benefits:
Preserves realistic processing variability
Reduces statistical distortion
Stabilizes regression models
Maintains full dataset integrity

 **Module 3 — ML **
 
🎯 Objective

The objective of this phase was to develop a predictive machine learning model capable of estimating visa processing timelines using historical PERM application data.

The model aims to:

Predict visa processing duration

Capture backlog and workload patterns

Identify influential features affecting processing delays

Evaluate model performance using statistical error metrics

🧠 Modeling Strategy

The modeling pipeline followed a structured machine learning workflow:

1️⃣ Feature Selection
2️⃣ Train–Test Split
3️⃣ Model Training
4️⃣ Model Evaluation
5️⃣ Model Comparison
6️⃣ Hyperparameter Optimization

This approach ensures robust, reproducible model performance evaluation.

📊 Feature Selection

After preprocessing and feature engineering, the following features were used for training:

📅 Time-Based Features

filing_year
filing_month
filing_quarter

📈 Workload & Demand Signals
monthly_volume
country_frequency

🏢 Employer Attributes
employer_city
employer_state
employer_name

📄 Case Information
class_of_admission
job_info_education
pw_job_title_9089

⚙ Train–Test Split

The dataset was divided into:
80% Training Data
20% Testing Data
This ensures the model is evaluated on unseen data for reliable performance measurement.

Models Evaluated

Three machine learning algorithms were implemented and compared:

Model	Purpose
Linear Regression -    	    Baseline regression model
Decision Tree Regressor - 	Tree-based nonlinear model
Random Forest Regressor	-   Ensemble learning model

Tree-based models were included because visa processing patterns are often nonlinear and influenced by multiple interacting factors.

Model Evaluation Metrics

Two standard regression metrics were used:

Mean Squared Error (MSE)
Measures the average squared difference between predicted and actual values.
Root Mean Squared Error (RMSE)
Represents the average prediction error in days, making the results easier to interpret.

Model Performance Comparison
Model	MSE,	RMSE (Days)
Linear Regression	9606 ,	98
Decision Tree	11521	, 107
Random Forest	7936	, 89
🏆 Best Model

Random Forest Regressor

The Random Forest model achieved the lowest prediction error, indicating stronger predictive capability compared to the other models.

📊 Model Evaluation Visualizations

Several visual diagnostics were used to evaluate model performance:

🔹 Model Error Comparison

Bar charts comparing MSE and RMSE across models.

🔹 Actual vs Predicted Analysis

Scatter plots showing the relationship between actual processing times and predicted values.

🔹 Residual Error Distribution

Histogram analysis confirming that prediction errors are centered around zero.

🔧 Hyperparameter Tuning

To further improve model performance, GridSearchCV was used to tune the Random Forest model.

Parameters explored included:
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
