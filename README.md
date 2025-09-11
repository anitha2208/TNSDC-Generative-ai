import pandas as pd
import json
import os
import glob
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


def description(file_path=None, chunk_size=10000):
    """
    If file_path is given → analyzes that CSV.
    If not given → picks the latest CSV from uploads/ folder.
    """
    if file_path is None:
        csv_files = glob.glob("uploads/*.csv")
        if not csv_files:
            return {"error": "No CSV files found in uploads/"}
        # pick latest uploaded CSV
        file_path = max(csv_files, key=os.path.getctime)

    analyzer = CSVAnalyzer(chunk_size=chunk_size)
    if not analyzer.read_csv_chunked(file_path):
        return {"error": f"Could not read file {file_path}"}
    analyzer.analyze_dataset()
    return analyzer.generate_json(job_id="description_call", file_path=file_path)


# Example usage (you can remove this part if integrating elsewhere)
if __name__ == "__main__":
    result = description()  # will auto-pick latest CSV in uploads/ if no file passed
    print(json.dumps(result, indent=4))
