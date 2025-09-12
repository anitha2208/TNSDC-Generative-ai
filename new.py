import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
import uuid
import shutil

# ------------------ analyze_csv ------------------
def analyze_csv(csv_path, chunk_size=10000):
    """
    Perform full analysis like report.json:
      - summary stats
      - missing values
      - numerical summary
      - categorical insights
      - data quality
      - data type separation
    """
    # --- Read file ---
    total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1

    if total_rows <= chunk_size:
        df = pd.read_csv(csv_path)
        processing_method = "full_load"
    else:
        df = pd.read_csv(csv_path, nrows=chunk_size)
        processing_method = "chunked"

    # Try convert object to datetime where possible
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    # Separate types
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    dt_cols = list(df.select_dtypes(include=["datetime64[ns]"]).columns)

    # Missing values
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df)) * 100
    total_missing = int(missing_data.sum())

    # Numerical summary
    numerical_summary = {}
    if num_cols:
        desc = df[num_cols].describe().to_dict()
        for col, stats in desc.items():
            numerical_summary[col] = {k: float(v) for k, v in stats.items()}

    # Categorical insights
    categorical_insights = {}
    for col in cat_cols:
        unique_values = df[col].nunique(dropna=False)
        most_common = df[col].value_counts(dropna=False).head(3).to_dict()
        categorical_insights[col] = {
            "unique_values": int(unique_values),
            "most_common_values": {str(k): int(v) for k, v in most_common.items()}
        }

    # Data quality
    duplicate_rows = int(df.duplicated().sum())
    total_cells = len(df) * len(df.columns)
    completeness_score = ((total_cells - total_missing) / total_cells) * 100 if total_cells else 0.0

    # Build JSON
    analysis = {
        "main_title": "CSV Dataset Overview & Analysis",
        "summary_stats": {
            "total_records": total_rows,
            "total_variables": len(df.columns),
            "completeness_percentage": round(completeness_score, 1),
            "processing_method": processing_method
        },
        "missing_values": {
            "total_missing": total_missing,
            "columns_with_missing": {k: int(v) for k, v in missing_data[missing_data > 0].to_dict().items()},
            "missing_percentages": {k: float(v) for k, v in missing_percentages[missing_percentages > 0].to_dict().items()}
        },
        "numerical_summary": numerical_summary,
        "categorical_insights": categorical_insights,
        "data_quality": {
            "duplicate_rows": duplicate_rows,
            "completeness_score": round(completeness_score, 1)
        },
        "data_type_separation": {
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "datetime_columns": dt_cols
        },
       
    }

    return analysis, num_cols, cat_cols, dt_cols

# ------------------ process_uploaded_csv ------------------
def process_uploaded_csv(csv_path, chunk_size=10000):
    """
    Importable function:
    - Takes path to a CSV file.
    - Creates unique ID folder under uploads and outputs.
    - Performs full analysis like report.json.
    - Returns a dictionary with uuid, JSON path, and separated columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    uploads_dir = Path("uploads")
    outputs_dir = Path("outputs")
    uploads_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    unique_id = str(uuid.uuid4().int)[:8]
    job_upload_dir = uploads_dir / unique_id
    job_upload_dir.mkdir(exist_ok=True)

    new_csv_path = job_upload_dir / csv_path.name
    if csv_path != new_csv_path:
        shutil.copy(csv_path, new_csv_path)

    analysis, num_cols, cat_cols, dt_cols = analyze_csv(new_csv_path, chunk_size)

    job_output_dir = outputs_dir / unique_id
    job_output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    json_path = job_output_dir / f"description_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    return {
        "uuid": unique_id,
        "json_path": str(json_path),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "datetime_columns": dt_cols
    }

# ------------------ main() for CLI ------------------
def main():
    if len(sys.argv) < 2:
        print("⚠️ No file uploaded. Usage: python new.py <csv_file>")
        return

    csv_path = Path(sys.argv[1])
    try:
        result = process_uploaded_csv(csv_path)
    except FileNotFoundError as e:
        print(e)
        return

    print(" Analysis complete.")
    print("UUID:", result["uuid"])
    print("JSON path:", result["json_path"])
    print("Numerical:", result["numerical_columns"])
    print("Categorical:", result["categorical_columns"])
    print("Datetime:", result["datetime_columns"])

if __name__ == "__main__":
    main()
