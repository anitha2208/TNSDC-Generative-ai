from new import process_uploaded_csv
result = process_uploaded_csv("uploads/BoTNeTIoT-L01-v2.csv")
print("UUID:", result["uuid"])
print("JSON Path:", result["json_path"])
print("Numerical Columns:", result["numerical_columns"])
print("Categorical Columns:", result["categorical_columns"])
print("Datetime Columns:", result["datetime_columns"])