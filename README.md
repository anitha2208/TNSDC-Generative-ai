import pandas as pd
import json
import os
from datetime import datetime


class CSVAnalyzer:
    def __init__(self, chunk_size=10000):
        self.chunk_size = chunk_size
        self.dataframe = None
        self.analysis = {}

    def read_csv_chunked(self, file_path):
        try:
            chunks = pd.read_csv(file_path, chunksize=self.chunk_size)
            self.dataframe = pd.concat(chunks, ignore_index=True)
            return True
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False

    def analyze_dataset(self):
        if self.dataframe is not None:
            self.analysis["rows"] = self.dataframe.shape[0]
            self.analysis["columns"] = self.dataframe.shape[1]
            self.analysis["column_names"] = list(self.dataframe.columns)
            self.analysis["dtypes"] = self.dataframe.dtypes.apply(lambda x: str(x)).to_dict()
            self.analysis["null_counts"] = self.dataframe.isnull().sum().to_dict()
        else:
            self.analysis = {"error": "No dataframe loaded"}

    def generate_json(self, job_id="job_001", file_path="unknown"):
        return {
            "job_id": job_id,
            "file_path": file_path,
            "processing_time": datetime.now().isoformat(),
            "chunk_size": self.chunk_size,
            "analysis": self.analysis
        }


# 🔹 Run the analysis when a file is uploaded
def run_analysis(input_file="uploads/input.csv", output_file="outputs/description.json", chunk_size=10000):
    if not os.path.exists(input_file):
        return {"status": "watching for uploads"}  # No file uploaded yet

    analyzer = CSVAnalyzer(chunk_size=chunk_size)
    if not analyzer.read_csv_chunked(input_file):
        return {"error": f"Could not read file {input_file}"}

    analyzer.analyze_dataset()
    result = analyzer.generate_json(job_id="description_call", file_path=input_file)

    # Save JSON output
    os.makedirs("outputs", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

    return result


# 🔹 Function others will call to get stored analysis
def description(output_file="outputs/description.json"):
    if not os.path.exists(output_file):
        return {"error": "No analysis found. Please upload a CSV first."}
    with open(output_file, "r") as f:
        return json.load(f)


# 🔹 Function to return numerical, categorical, datetime columns
def get_column_types(output_file="outputs/description.json"):
    if not os.path.exists(output_file):
        return [], [], []  # no analysis yet

    with open(output_file, "r") as f:
        analysis = json.load(f)

    dtypes = analysis.get("analysis", {}).get("dtypes", {})

    numerical = [col for col, dtype in dtypes.items() if dtype in ["int64", "float64"]]
    categorical = [col for col, dtype in dtypes.items() if dtype == "object"]
    datetime_cols = [col for col, dtype in dtypes.items() if "datetime" in dtype]

    return numerical, categorical, datetime_cols


# 🔹 Example flow (for testing locally)
if __name__ == "__main__":
    print("🔄 Running analysis...")
    run_result = run_analysis()
    print(json.dumps(run_result, indent=4))

    print("\n📌 Stored description:")
    print(json.dumps(description(), indent=4))

    print("\n📊 Column types (num, cat, datetime):")
    print(get_column_types())
