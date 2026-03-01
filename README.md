# AI-Powered-Visa-Delay-Forecasting-Approval-Risk-Intelligence-Platform
The AI-Powered Visa Delay Forecasting & Approval Risk Intelligence Platform is a cutting-edge solution designed to demystify the complex world of global immigration. By leveraging advanced machine learning algorithms, this platform provides applicants, legal firms, and HR departments with high-accuracy predictions for visa processing times and a comprehensive risk assessment of application approval. 
It analyzes historical trends, current regulatory environments, and individual application data to identify potential red flags before submission, significantly reducing the uncertainty of international mobility and improving success rates through proactive, data-driven intelligence.

## Dataset Cleaning & Preprocessing (Module 1)

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

