import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Directory configuration for worker loop
UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")

# Create directories if they don't exist
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

class ExcelToJSONAnalyzer:
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
                print(f"✅ Successfully loaded CSV file with {len(self.df)} rows and {len(self.df.columns)} columns")
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path)
                print(f"✅ Successfully loaded Excel file with {len(self.df)} rows and {len(self.df.columns)} columns")
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")

            return True
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
            return False

    def read_streamlit_file(self, job_id):
        """Read file from Streamlit uploads using job ID"""
        try:
            # Look for files in the job directory
            job_dir = UPLOADS_DIR / job_id
            if not job_dir.exists():
                raise FileNotFoundError(f"Job directory {job_dir} not found")
            
            # Get the first file in the directory
            files = list(job_dir.glob("*"))
            if not files:
                raise FileNotFoundError(f"No files found in {job_dir}")
            
            file_path = files[0]  # Take the first file
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.csv':
                self.df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path)
            else:
                # Try to read as CSV anyway
                self.df = pd.read_csv(file_path)
            
            self.file_path = str(file_path)
            print(f"✅ Successfully loaded file {file_path.name} from job {job_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error reading file for job {job_id}: {str(e)}")
            return False

    def separate_data_types(self):
        """Separate data into categorical and numerical columns and return as tuple"""
        if self.df is None:
            return ([], [])
        
        numerical_columns = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_columns = list(self.df.select_dtypes(include=['object']).columns)
        
        return (categorical_columns, numerical_columns)

    def analyze_dataset(self):
        """Comprehensive data analysis for JSON output"""
        if self.df is None:
            return None

        analysis = {}

        # Basic dataset information
        analysis['basic_info'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'file_name': os.path.basename(self.file_path),
            'columns': list(self.df.columns)
        }

        # Data types analysis
        categorical_cols, numerical_cols = self.separate_data_types()
        analysis['data_types'] = {
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': list(self.df.select_dtypes(include=['datetime64']).columns)
        }

        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_percentages = (missing_data / len(self.df)) * 100

        analysis['missing_values'] = {
            'total_missing': int(missing_data.sum()),
            'columns_with_missing': {k: int(v) for k, v in missing_data[missing_data > 0].to_dict().items()},
            'missing_percentages': {k: float(v) for k, v in missing_percentages[missing_percentages > 0].to_dict().items()}
        }

        # Statistical summary for numerical columns
        if analysis['data_types']['numerical_columns']:
            numerical_summary = self.df[analysis['data_types']['numerical_columns']].describe()
            analysis['numerical_summary'] = {k: {k2: float(v2) if not pd.isna(v2) else None for k2, v2 in v.items()} for k, v in numerical_summary.to_dict().items()}

        # Categorical data insights
        categorical_insights = {}
        for col in analysis['data_types']['categorical_columns']:
            unique_values = self.df[col].nunique()
            most_common = self.df[col].value_counts().head(3).to_dict()
            categorical_insights[col] = {
                'unique_values': int(unique_values),
                'most_common_values': {k: int(v) for k, v in most_common.items()}
            }
        analysis['categorical_insights'] = categorical_insights

        # Data quality assessment
        duplicate_rows = self.df.duplicated().sum()
        analysis['data_quality'] = {
            'duplicate_rows': int(duplicate_rows),
            'completeness_score': float(((len(self.df) * len(self.df.columns) - missing_data.sum()) / 
                                 (len(self.df) * len(self.df.columns))) * 100)
        }

        self.analysis_results = analysis
        return analysis

    def generate_dataset_title(self):
        """Generate intelligent title based on dataset analysis"""
        if not self.analysis_results:
            return "Dataset Analysis Report"

        file_name = self.analysis_results['basic_info']['file_name']
        
        # Extract base name without extension
        base_name = os.path.splitext(file_name)[0].replace('_', ' ').replace('-', ' ').title()

        # Create intelligent title based on content
        categorical_cols = self.analysis_results['data_types']['categorical_columns']
        numerical_cols = self.analysis_results['data_types']['numerical_columns']

        if 'sales' in file_name.lower() or any('sales' in col.lower() for col in self.df.columns):
            title = f"Sales Data Analysis: {base_name}"
        elif 'customer' in file_name.lower() or any('customer' in col.lower() for col in self.df.columns):
            title = f"Customer Data Insights: {base_name}"
        elif 'employee' in file_name.lower() or any('employee' in col.lower() for col in self.df.columns):
            title = f"Employee Data Analysis: {base_name}"
        elif len(numerical_cols) > len(categorical_cols):
            title = f"Numerical Data Analysis: {base_name}"
        else:
            title = f"Data Analysis Report: {base_name}"

        return title

    def generate_left_column_content(self):
        """Generate content for left column"""
        if not self.analysis_results:
            return "No analysis available"

        analysis = self.analysis_results
        content = []

        # Dataset Overview
        content.append("• Dataset Overview")
        content.append(f"  - {analysis['basic_info']['total_rows']:,} total records")
        content.append(f"  - {analysis['basic_info']['total_columns']} feature variables")
        content.append(f"  - {len(analysis['data_types']['numerical_columns'])} numerical fields")
        content.append(f"  - {len(analysis['data_types']['categorical_columns'])} categorical fields")
        content.append("")

        # Data Quality
        content.append("• Data Quality Assessment")
        completeness = analysis['data_quality']['completeness_score']
        content.append(f"  - Overall completeness: {completeness:.1f}%")

        if analysis['data_quality']['duplicate_rows'] > 0:
            content.append(f"  - Duplicate records: {analysis['data_quality']['duplicate_rows']}")
        else:
            content.append(f"  - No duplicate records detected")

        quality_rating = "Excellent" if completeness > 95 else "Good" if completeness > 85 else "Fair" if completeness > 70 else "Poor"
        content.append(f"  - Data quality rating: {quality_rating}")
        content.append("")

        # Missing Data Impact
        content.append("• Missing Data Impact")
        if analysis['missing_values']['total_missing'] > 0:
            missing_cols = analysis['missing_values']['columns_with_missing']
            content.append(f"  - {len(missing_cols)} columns affected")
            content.append(f"  - {analysis['missing_values']['total_missing']:,} missing values total")

            # Show top 2 columns with missing data
            sorted_missing = sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:2]
            for col, missing_count in sorted_missing:
                percentage = analysis['missing_values']['missing_percentages'][col]
                content.append(f"  - {col}: {percentage:.1f}% missing")
        else:
            content.append("  - No missing values detected")
            content.append("  - Complete data integrity maintained")

        return "\n".join(content)

    def generate_right_column_content(self):
        """Generate content for right column"""
        if not self.analysis_results:
            return "No analysis available"

        analysis = self.analysis_results
        content = []

        # Analytical Capabilities
        content.append("• Analytical Capabilities")
        numerical_cols = len(analysis['data_types']['numerical_columns'])
        categorical_cols = len(analysis['data_types']['categorical_columns'])

        if numerical_cols > 0:
            content.append(f"  - Statistical analysis ready")
            content.append(f"  - Trend and correlation studies")
            content.append(f"  - Predictive modeling potential")

        if categorical_cols > 0:
            content.append(f"  - Segmentation analysis available")
            content.append(f"  - Classification studies possible")

        if numerical_cols > 0 and categorical_cols > 0:
            content.append(f"  - Mixed-type analysis supported")

        content.append("")

        # Business Value
        content.append("• Business Value Indicators")

        # Data volume assessment
        data_volume = analysis['basic_info']['total_rows']
        if data_volume > 10000:
            content.append("  - Large-scale dataset for enterprise insights")
        elif data_volume > 1000:
            content.append("  - Medium-scale dataset for departmental analysis")
        else:
            content.append("  - Focused dataset for targeted analysis")

        # Variable richness
        var_count = analysis['basic_info']['total_columns']
        if var_count > 20:
            content.append("  - Rich feature set for comprehensive analysis")
        elif var_count > 10:
            content.append("  - Adequate variables for detailed insights")
        else:
            content.append("  - Focused variable set for specific analysis")

        # Data quality business impact
        completeness = analysis['data_quality']['completeness_score']
        if completeness > 90:
            content.append("  - High reliability for business decisions")
        elif completeness > 75:
            content.append("  - Moderate reliability with some limitations")
        else:
            content.append("  - Requires data cleaning for optimal use")

        content.append("")

        # Key Applications
        content.append("• Key Applications")

        # Determine applications based on data characteristics
        file_name = analysis['basic_info']['file_name'].lower()
        columns = [col.lower() for col in analysis['basic_info']['columns']]

        if any(keyword in file_name or any(keyword in col for col in columns) 
               for keyword in ['sales', 'revenue', 'price', 'cost']):
            content.append("  - Sales performance analysis")
            content.append("  - Revenue optimization studies")
            content.append("  - Pricing strategy development")
        elif any(keyword in file_name or any(keyword in col for col in columns) 
                 for keyword in ['customer', 'client', 'user']):
            content.append("  - Customer behavior analysis")
            content.append("  - Market segmentation studies")
            content.append("  - Customer lifetime value modeling")
        elif any(keyword in file_name or any(keyword in col for col in columns) 
                 for keyword in ['employee', 'staff', 'hr', 'payroll']):
            content.append("  - Workforce analytics")
            content.append("  - Performance management insights")
            content.append("  - HR optimization strategies")
        else:
            content.append("  - Exploratory data analysis")
            content.append("  - Pattern recognition studies")
            content.append("  - Statistical modeling projects")
            content.append("  - Business intelligence reporting")

        return "\n".join(content)

    def generate_json_output(self):
        """Generate JSON output with analysis and formatted content"""
        if self.df is None or not self.analysis_results:
            return {"error": "No data available for analysis"}

        analysis = self.analysis_results
        completeness = analysis['data_quality']['completeness_score']

        # Get data type separation as tuple
        data_type_tuple = self.separate_data_types()

        json_output = {
            "metadata": {
                "title": self.generate_dataset_title(),
                "generated_on": datetime.now().strftime('%B %d, %Y'),
                "summary_stats": f"{analysis['basic_info']['total_rows']:,} Records • {analysis['basic_info']['total_columns']} Variables • {completeness:.1f}% Complete"
            },
            "data_separation": {
                "categorical_numerical_tuple": data_type_tuple,
                "categorical_columns": data_type_tuple[0],
                "numerical_columns": data_type_tuple[1]
            },
            "content_divisions": {
                "left_division": self.generate_left_column_content(),
                "right_division": self.generate_right_column_content()
            },
            "detailed_analysis": self.analysis_results,
            "processing_info": {
                "file_source": os.path.basename(self.file_path) if self.file_path else "Unknown",
                "processing_timestamp": datetime.now().isoformat()
            }
        }

        return json_output

    def save_json_output(self, job_id, json_data):
        """Save JSON output to the outputs directory"""
        try:
            output_dir = OUTPUTS_DIR / job_id
            output_dir.mkdir(exist_ok=True)
            
            json_path = output_dir / "analysis.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON analysis saved to {json_path}")
            return str(json_path)
        except Exception as e:
            print(f"❌ Error saving JSON output: {str(e)}")
            return None

    def process_file_streamlit(self, job_id=None, file_path=None):
        """Main processing function for Streamlit integration"""
        print(f"🚀 Starting analysis for job {job_id}...")
        print("=" * 50)

        # Step 1: Read file
        if job_id:
            print(f"📂 Step 1: Reading file for job {job_id}...")
            if not self.read_streamlit_file(job_id):
                return {"error": "Failed to read file"}
        elif file_path:
            print(f"📂 Step 1: Reading file {file_path}...")
            if not self.read_file(file_path):
                return {"error": "Failed to read file"}
        else:
            return {"error": "No job ID or file path provided"}

        # Step 2: Display basic file info
        print(f"📋 File Info:")
        print(f"   - File: {os.path.basename(self.file_path)}")
        print(f"   - Dimensions: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"   - Columns: {', '.join(self.df.columns.tolist())}")

        # Step 3: Analyze dataset
        print("\n🔍 Step 2: Analyzing dataset...")
        self.analyze_dataset()

        # Step 4: Generate JSON output
        print("\n📊 Step 3: Creating JSON analysis...")
        json_result = self.generate_json_output()

        # Step 5: Save JSON output if job_id provided
        if job_id:
            print("\n💾 Step 4: Saving JSON output...")
            json_path = self.save_json_output(job_id, json_result)
            if json_path:
                json_result["output_path"] = json_path

        print(f"\n✅ Analysis complete for job {job_id}!")
        return json_result


def analyze_csv_to_json(job_id=None, file_path=None):
    """Standalone function for easy integration with Streamlit"""
    analyzer = ExcelToJSONAnalyzer()
    return analyzer.process_file_streamlit(job_id=job_id, file_path=file_path)


def make_sample_analysis(job_id, input_file):
    """Generate JSON analysis for a specific job - replacement for make_sample_ppt"""
    try:
        print(f"🎯 Processing job {job_id} with file {input_file}")
        
        # Create analyzer and process file
        analyzer = ExcelToJSONAnalyzer()
        
        # Read the file directly
        if not analyzer.read_file(input_file):
            print(f"❌ Failed to process {input_file} for job {job_id}")
            return False
        
        # Analyze dataset
        analyzer.analyze_dataset()
        
        # Generate JSON output
        json_result = analyzer.generate_json_output()
        
        # Save to outputs directory
        json_path = analyzer.save_json_output(job_id, json_result)
        
        if json_path:
            print(f"✅ Analysis completed for job {job_id}")
            return True
        else:
            print(f"❌ Failed to save analysis for job {job_id}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing job {job_id}: {str(e)}")
        return False


def worker_loop(poll_interval=5):
    """Worker loop to process uploaded files and generate JSON analyses"""
    print("🔄 Worker started, watching for uploads...")
    
    while True:
        try:
            # Check all job directories in uploads
            for job_dir in UPLOADS_DIR.glob("*"):
                if not job_dir.is_dir():
                    continue
                    
                job_id = job_dir.name
                output_dir = OUTPUTS_DIR / job_id
                analysis_path = output_dir / "analysis.json"
                
                # Skip if analysis already exists
                if analysis_path.exists():
                    continue
                
                # Get files in job directory
                files = list(job_dir.glob("*"))
                if not files:
                    continue
                
                # Process the first file found
                input_file = str(files[0])
                print(f"📂 Found new job: {job_id}")
                
                # Generate analysis instead of PPT
                make_sample_analysis(job_id, input_file)
                
        except Exception as e:
            print(f"❌ Error in worker loop: {str(e)}")
            
        print(f"💤 Sleeping for {poll_interval} seconds...")
        time.sleep(poll_interval)


# Example usage functions
def main():
    """Main function to run the analyzer"""
    analyzer = ExcelToJSONAnalyzer()

    # Example 1: Using job ID from Streamlit
    print("Example 1: Using job ID")
    result1 = analyzer.process_file_streamlit(job_id="job_123")
    print(json.dumps(result1, indent=2))

    # Example 2: Using direct file path
    print("\nExample 2: Using file path")
    result2 = analyzer.process_file_streamlit(file_path="input.csv")
    print(json.dumps(result2, indent=2))


if __name__ == "__main__":
    print("🎯 Excel to JSON Analyzer Worker")
    print("=" * 40)
    print("Supported formats: CSV, XLSX, XLS")
    print("Directory structure:")
    print(f"  - Uploads: {UPLOADS_DIR}")
    print(f"  - Outputs: {OUTPUTS_DIR}")
    print("=" * 40)

    try:
        # Start the worker loop
        worker_loop()
    except KeyboardInterrupt:
        print("\n❌ Worker stopped by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

    print("\n👆 Worker finished.")