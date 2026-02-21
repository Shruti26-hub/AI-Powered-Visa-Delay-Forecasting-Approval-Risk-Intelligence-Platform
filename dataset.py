import pandas as pd

# ==========================================
# STEP 1: GIVE CORRECT FILE PATH
# ==========================================

excel_file = r"C:/Users/Gapat/OneDrive/Desktop/visa/cleaned visa dataset.xlsx"  # <-- UPDATE THIS PATH TO YOUR EXCEL FILE

# ==========================================
# STEP 2: LOAD EXCEL FILE (NOT CSV)
# ==========================================

df = pd.read_excel(excel_file)

print(" Excel Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# ==========================================
# STEP 3: CHECK COLUMN NAMES (IMPORTANT)
# ==========================================

print("\nColumns in Dataset:")
print(df.columns)

# ==========================================
# STEP 4: REMOVE ROWS WITH MISSING DATES
# ==========================================

df = df.dropna(subset=["case_received_date", "decision_date"])

print("\n After Removing Missing Dates:", df.shape)

# ==========================================
# STEP 5: CONVERT DATE COLUMNS TO DATETIME
# ==========================================

df["case_received_date"] = pd.to_datetime(df["case_received_date"], errors="coerce")
df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")

df = df.dropna(subset=["case_received_date", "decision_date"])

print(" After Date Conversion:", df.shape)

# ==========================================
# STEP 6: REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print(" After Removing Duplicates:", df.shape)

# ==========================================
# STEP 7: CREATE PROCESSING DAYS FEATURE
# ==========================================

df["processing_days"] = (
    df["decision_date"] - df["case_received_date"]
).dt.days

# Remove negative values
df = df[df["processing_days"] >= 0]

print(" After Adding Processing Days:", df.shape)

# ==========================================
# STEP 8: SAVE CLEAN DATASET AS CSV
# ==========================================

output_csv = r"C:\Users\Gapat\Downloads\perm_final_clean.csv"

df.to_csv(output_csv, index=False)

print("\n DONE! Clean dataset saved successfully.")
print(" Output File Location:", output_csv)


