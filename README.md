# AI-Powered-Visa-Delay-Forecasting-Approval-Risk-Intelligence-Platform
The AI-Powered Visa Delay Forecasting & Approval Risk Intelligence Platform is a cutting-edge solution designed to demystify the complex world of global immigration. By leveraging advanced machine learning algorithms, this platform provides applicants, legal firms, and HR departments with high-accuracy predictions for visa processing times and a comprehensive risk assessment of application approval. 
It analyzes historical trends, current regulatory environments, and individual application data to identify potential red flags before submission, significantly reducing the uncertainty of international mobility and improving success rates through proactive, data-driven intelligence.

## Dataset Cleaning & Preprocessing (Module 1)

---

##  Project Overview

Visa applicants often face uncertainty due to long and unpredictable processing times.  
This project aims to build a predictive analytics system that estimates visa processing time and supports visa status prediction using historical PERM visa application data.

This README documents the complete **data preprocessing and cleaning pipeline

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

