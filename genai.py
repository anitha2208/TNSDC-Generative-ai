import pandas as pd
import numpy as np
import json
import sys
import time
from datetime import datetime
from pathlib import Path


class CSVAnalyzer:
    def __init__(self, chunk_size=10000):
        self.chunk_size = chunk_size
        self.df = None
        self.file_path = None
        self.total_rows = 0
        self.columns = []

    def read_csv(self, file_path):
        """Read CSV (chunked if large)."""
        self.file_path = file_path
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.total_rows = sum(1 for _ in f) - 1

            if self.total_rows <= self.chunk_size:
                self.df = pd.read_csv(file_path)
            else:
                # For large files, just read first chunk but mark as chunked
                self.df = pd.read_csv(file_path, nrows=self.chunk_size)

            self.columns = list(self.df.columns)
            return True
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False

    def analyze(self):
        """Comprehensive dataset analysis."""
        if self.df is None:
            return {"error": "No data loaded"}

        # Separate data types
        num_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(self.df.select_dtypes(include=["object", "category"]).columns)
        dt_cols = list(self.df.select_dtypes(include=["datetime64"]).columns)

        # Missing values
        missing_data = self.df.isnull().sum()
        missing_percentages = (missing_data / len(self.df)) * 100
        total_missing = int(missing_data.sum())

        # Numerical summary
        numerical_summary = {}
        if num_cols:
            summary = self.df[num_cols].describe().to_dict()
            for col, stats in summary.items():
                numerical_summary[col] = {k: float(v) for k, v in stats.items()}

        # Categorical insights
        categorical_insights = {}
        for col in cat_cols:
            unique_values = self.df[col].nunique()
            most_common = self.df[col].value_counts().head(3).to_dict()
            categorical_insights[col] = {
                "unique_values": int(unique_values),
                "most_common_values": {str(k): int(v) for k, v in most_common.items()}
            }

        # Data quality
        duplicate_rows = int(self.df.duplicated().sum())
        total_cells = len(self.df) * len(self.df.columns)
        completeness_score = ((total_cells - total_missing) / total_cells) * 100

        # JSON structure
        return {
            "main_title": "CSV Dataset Overview & Analysis",
            "summary_stats": {
                "total_records": self.total_rows,
                "total_variables": len(self.columns),
                "completeness_percentage": round(completeness_score, 1),
                "processing_method": "chunked" if self.total_rows > self.chunk_size else "full_load"
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
            "generation_timestamp": datetime.now().isoformat(),
            "processing_info": {
                "file_path": self.file_path,
                "chunk_size_used": self.chunk_size,
                "processing_status": "completed",
            },
        }


def process_csv(file_path, chunk_size=10000):
    analyzer = CSVAnalyzer(chunk_size=chunk_size)
    if analyzer.read_csv(file_path):
        return analyzer.analyze()
    return {"error": f"Failed to process {file_path}"}


def worker_loop(chunk_size=10000, poll_interval=5):
    """Worker loop: watch 'uploads' for job folders with CSVs inside."""
    uploads = Path("uploads")
    outputs = Path("outputs")
    uploads.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)

    print("🔄 Worker started. Watching 'uploads/<job_id>/input.csv' ...")

    while True:
        for job_dir in uploads.glob("*"):  # each subfolder is a job_id
            if not job_dir.is_dir():
                continue

            job_id = job_dir.name
            out_dir = outputs / job_id
            out_dir.mkdir(exist_ok=True)

            json_path = out_dir / "report.json"
            if json_path.exists():
                continue  # already processed

            # Look for a CSV inside the job folder
            csv_files = list(job_dir.glob("*.csv"))
            if not csv_files:
                continue

            csv_file = csv_files[0]  # take the first CSV
            print(f"📂 Found job: {job_id}, processing {csv_file.name}")

            result = process_csv(csv_file, chunk_size)

            with open(json_path, "w") as f:
                json.dump(result, f, indent=2, default=str)  # default=str for safety

            print(f"✅ Job {job_id} complete → {json_path}")

        time.sleep(poll_interval)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) < 3:
            print("Usage: python simple.py test <csv_file>")
            return

        file_path = sys.argv[2]
        result = process_csv(file_path)

        Path("outputs").mkdir(exist_ok=True)
        out_file = Path("outputs") / "report.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        print("✅ Analysis complete. JSON saved to outputs/report.json")
    else:
        worker_loop()


if __name__ == "__main__":
    main()
