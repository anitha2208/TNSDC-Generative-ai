import pandas as pd
import numpy as np
import os
import sys
import platform
import subprocess
import uuid
import json
from datetime import datetime
import time
from pathlib import Path
import pandas as pd
from pptx import Presentation


class ExcelToJSONConverter:
    def __init__(self):
        self.df = None
        self.file_path = None
        self.analysis_results = {}

    def read_file(self, file_path):
        """Read CSV or XLSX files"""
        self.file_path = file_path
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == '.csv':
                self.df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            return True
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
            return False

    def analyze_dataset(self):
        """Comprehensive data analysis for JSON insights"""
        if self.df is None:
            return None

        analysis = {}
        analysis['basic_info'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'file_name': os.path.basename(self.file_path),
            'columns': list(self.df.columns)
        }

        # Separate numerical and categorical columns
        numerical_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(self.df.select_dtypes(include=['object']).columns)

        analysis['data_types'] = {
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': list(self.df.select_dtypes(include=['datetime64']).columns)
        }

        # Missing values
        missing_data = self.df.isnull().sum()
        missing_percentages = (missing_data / len(self.df)) * 100
        analysis['missing_values'] = {
            'total_missing': int(missing_data.sum()),
            'columns_with_missing': missing_data[missing_data > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
        }

        # Numerical summary
        if numerical_cols:
            numerical_summary = self.df[numerical_cols].describe()
            analysis['numerical_summary'] = numerical_summary.to_dict()

        # Categorical insights
        categorical_insights = {}
        for col in categorical_cols:
            unique_values = self.df[col].nunique()
            most_common = self.df[col].value_counts().head(3).to_dict()
            categorical_insights[col] = {
                'unique_values': int(unique_values),
                'most_common_values': most_common
            }
        analysis['categorical_insights'] = categorical_insights

        # Data quality
        duplicate_rows = self.df.duplicated().sum()
        analysis['data_quality'] = {
            'duplicate_rows': int(duplicate_rows),
            'completeness_score': float(((len(self.df) * len(self.df.columns) - missing_data.sum()) / 
                                  (len(self.df) * len(self.df.columns))) * 100)
        }

        self.analysis_results = analysis
        return analysis

    def generate_left_column_content(self):
        """Left column text for JSON"""
        analysis = self.analysis_results
        content = []

        content.append("Dataset Overview")
        content.append(f"Total Records: {analysis['basic_info']['total_rows']}")
        content.append(f"Total Variables: {analysis['basic_info']['total_columns']}")
        content.append(f"Numerical Fields: {len(analysis['data_types']['numerical_columns'])}")
        content.append(f"Categorical Fields: {len(analysis['data_types']['categorical_columns'])}")

        completeness = analysis['data_quality']['completeness_score']
        quality_rating = "Excellent" if completeness > 95 else "Good" if completeness > 85 else "Fair" if completeness > 70 else "Poor"

        content.append("Data Quality")
        content.append(f"Completeness: {completeness:.1f}%")
        content.append(f"Duplicate Records: {analysis['data_quality']['duplicate_rows']}")
        content.append(f"Quality Rating: {quality_rating}")

        return content

    def generate_right_column_content(self):
        """Right column text for JSON"""
        analysis = self.analysis_results
        content = []

        content.append("Analytical Capabilities")
        if analysis['data_types']['numerical_columns']:
            content.append("Supports statistical and predictive modeling")
        if analysis['data_types']['categorical_columns']:
            content.append("Supports segmentation and classification")
        if analysis['data_types']['numerical_columns'] and analysis['data_types']['categorical_columns']:
            content.append("Supports mixed-type analysis")

        content.append("Business Value")
        rows = analysis['basic_info']['total_rows']
        cols = analysis['basic_info']['total_columns']
        if rows > 10000:
            content.append("Large dataset for enterprise insights")
        elif rows > 1000:
            content.append("Medium dataset for departmental analysis")
        else:
            content.append("Small dataset for targeted analysis")

        if cols > 20:
            content.append("Rich feature set for deep insights")
        elif cols > 10:
            content.append("Adequate variables for detailed analysis")
        else:
            content.append("Focused dataset for specific use cases")

        return content

    def generate_json_report(self):
        """Return final JSON structure"""
        if not self.analysis_results:
            return {}

        report = {
            "file_info": self.analysis_results['basic_info'],
            "left_column": self.generate_left_column_content(),
            "right_column": self.generate_right_column_content(),
            "separated_data": {
                "numerical": tuple(self.analysis_results['data_types']['numerical_columns']),
                "categorical": tuple(self.analysis_results['data_types']['categorical_columns'])
            },
            "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return report

    def process_file(self, file_path):
        """Main processing function"""
        if not self.read_file(file_path):
            return {}
        self.analyze_dataset()
        return self.generate_json_report()


# ---------------- STREAMLIT FRONTEND ----------------
# streamlit_app.py
import streamlit as st

def run_streamlit_ui():
    st.title("📊 Excel/CSV to JSON Converter")

    uploaded_file = st.file_uploader("Upload your CSV/Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        # Save with unique ID
        unique_id = str(uuid.uuid4())
        file_ext = os.path.splitext(uploaded_file.name)[1]
        save_path = f"uploads/{unique_id}_input{file_ext}"

        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File saved with ID: {unique_id}")

        # Process file
        converter = ExcelToJSONConverter()
        report = converter.process_file(save_path)

        st.json(report)


if __name__ == "__main__":
    import time
from pathlib import Path
from PIL import Image

# Directories
UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Dummy function to simulate processing
def make_sample_ppt(job_id, input_file):
    print(f"Processing job {job_id} with file {input_file}")
    # Example: open the image to prove it's working
    img = Image.open(input_file)
    print(f"Image size: {img.size}")
    # You can add your PPT creation logic here
    # (e.g., using python-pptx to insert this image in a slide)

def worker_loop(poll_interval=5):
    print("Worker started. Watching for uploads...")

    while True:
        # Check for uploaded jobs
        for job_dir in UPLOADS_DIR.glob("*"):
            job_id = job_dir.name
            out_dir = OUTPUTS_DIR / job_id
            ppt_path = out_dir / "presentation.pptx"

            if ppt_path.exists():
                continue  # already processed

            # Pick first file in job upload dir
            files = list(job_dir.glob("*"))
            if not files:
                continue

            input_file = str(files[0])
            make_sample_ppt(job_id, input_file)

        time.sleep(poll_interval)


if __name__ == "__main__":
    import time
from pathlib import Path
from PIL import Image

# Directories
UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Dummy function to simulate processing
def make_sample_ppt(job_id, input_file):
    print(f"Processing job {job_id} with file {input_file}")
    # Example: open the image to prove it's working
    img = Image.open(input_file)
    print(f"Image size: {img.size}")
    # You can add your PPT creation logic here
    # (e.g., using python-pptx to insert this image in a slide)

def worker_loop(poll_interval=5):
    print("Worker started. Watching for uploads...")

    while True:
        # Check for uploaded jobs
        for job_dir in UPLOADS_DIR.glob("*"):
            job_id = job_dir.name
            out_dir = OUTPUTS_DIR / job_id
            ppt_path = out_dir / "presentation.pptx"

            if ppt_path.exists():
                continue  # already processed

            # Pick first file in job upload dir
            files = list(job_dir.glob("*"))
            if not files:
                continue

            input_file = str(files[0])
            make_sample_ppt(job_id, input_file)

        time.sleep(poll_interval)


if __name__ == "__main__":
    import time
from pathlib import Path
from PIL import Image

# Directories
UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Dummy function to simulate processing
def make_sample_ppt(job_id, input_file):
    print(f"Processing job {job_id} with file {input_file}")
    # Example: open the image to prove it's working
    img = Image.open(input_file)
    print(f"Image size: {img.size}")
    # You can add your PPT creation logic here
    # (e.g., using python-pptx to insert this image in a slide)

def worker_loop(poll_interval=5):
    print("Worker started. Watching for uploads...")
    while True:
        for job_dir in UPLOADS_DIR.glob("*"):
            job_id = job_dir.name
            out_dir = OUTPUTS_DIR / job_id
            out_dir.mkdir(parents=True, exist_ok=True)   # ensure output folder exists
            
            ppt_path = out_dir / "presentation.pptx"
            json_path = out_dir / "report.json"

            if ppt_path.exists():
                continue  # already processed

            # Pick first file in job upload dir
            files = list(job_dir.glob("*.csv"))   # ensure only CSV files
            if not files:
                continue

            input_file = str(files[0])
            print(f"Processing {input_file}...")

            # Generate PPT
            make_sample_ppt(job_id, input_file)

            # (Optional) Save dummy JSON for testing
            with open(json_path, "w") as f:
                f.write('{"status": "success", "file": "' + input_file + '"}')

            print(f"✅ Output generated:")
            print(f"   PPT  -> {ppt_path}")
            print(f"   JSON -> {json_path}")

        time.sleep(poll_interval)


if __name__ == "__main__":
    worker_loop()
   
                

