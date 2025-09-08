import pandas as pd
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime

def read_csv_file(file_path):
    """Read CSV file with proper indexing for large files"""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # For large files, ensure proper indexing without data loss
        if len(df) > 10000:
            print(f"Large file detected ({len(df)} rows). Applying indexing...")
            df.reset_index(drop=True, inplace=True)
        
        print(f"Successfully loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def get_data_type_separation(df):
    """Separate data into categorical, numerical, and datetime columns - returns tuples"""
    if df is None:
        return (), (), ()
    
    # Numerical columns
    numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
    
    # Categorical columns  
    categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
    
    # DateTime columns
    datetime_columns = list(df.select_dtypes(include=['datetime64']).columns)
    
    # Try to detect datetime from object columns
    for col in categorical_columns.copy():
        try:
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                pd.to_datetime(sample, errors='raise')
                datetime_columns.append(col)
                categorical_columns.remove(col)
        except:
            continue
    
    return (
        tuple(numerical_columns),
        tuple(categorical_columns), 
        tuple(datetime_columns)
    )

def analyze_csv_data(df, file_name):
    """Analyze CSV data and generate the required JSON structure"""
    if df is None:
        return None
    
    # Get data type separation
    numerical_cols, categorical_cols, datetime_cols = get_data_type_separation(df)
    
    # Calculate missing values
    total_missing = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    completeness_percentage = ((total_cells - total_missing) / total_cells) * 100
    
    # Create analysis structure matching your requirements
    analysis = {
        "main_title": "Dataset Overview & Analysis",
        "summary_stats": {
            "total_records": len(df),
            "total_variables": len(df.columns),
            "completeness_percentage": round(completeness_percentage, 1)
        },
        "insights": {
            "content": f"• Analytical Capabilities\n - Statistical analysis ready\n - Trend and correlation studies\n - Predictive modeling potential\n - Segmentation analysis available\n - Classification studies possible\n - Mixed-type analysis supported\n\n• Business Value Indicators\n - {'Large-scale' if len(df) > 10000 else 'Medium-scale' if len(df) > 1000 else 'Small-scale'} dataset for analysis\n - Rich feature set for comprehensive analysis\n - {'High' if completeness_percentage > 90 else 'Medium' if completeness_percentage > 75 else 'Low'} reliability for business decisions\n\n• Key Applications\n - Customer behavior analysis\n - Market segmentation studies\n - Customer lifetime value modeling",
            "structured_data": {
                "analytical_capabilities": {
                    "statistical_analysis": len(numerical_cols) > 0,
                    "segmentation_analysis": len(categorical_cols) > 0,
                    "mixed_type_analysis": len(numerical_cols) > 0 and len(categorical_cols) > 0
                },
                "business_value": {
                    "data_scale": "Large-scale" if len(df) > 10000 else "Medium-scale" if len(df) > 1000 else "Small-scale",
                    "variable_richness": "Rich" if len(df.columns) > 20 else "Adequate" if len(df.columns) > 10 else "Focused",
                    "reliability": "High" if completeness_percentage > 90 else "Medium" if completeness_percentage > 75 else "Low"
                }
            }
        },
        "data_type_separation": {
            "numerical_columns": list(numerical_cols),
            "categorical_columns": list(categorical_cols),
            "datetime_columns": list(datetime_cols)
        },
        "full_analysis": {
            "basic_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "file_name": file_name,
                "columns": list(df.columns)
            },
            "data_types": {
                "numerical_columns": list(numerical_cols),
                "categorical_columns": list(categorical_cols),
                "datetime_columns": list(datetime_cols)
            },
            "missing_values": {
                "total_missing": int(total_missing),
                "columns_with_missing": {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0},
                "missing_percentages": {col: round((df[col].isnull().sum() / len(df)) * 100, 2) for col in df.columns if df[col].isnull().sum() > 0}
            }
        },
        "generation_timestamp": datetime.now().isoformat()
    }
    
    return analysis

def get_base_path():
    """Get the base path - works from any directory including VS Code"""
    # Get current working directory (where script is run from)
    current_dir = Path.cwd()
    
    # Check if we're in a subdirectory and need to go up
    script_dir = Path(__file__).parent.absolute()
    
    # Use script directory as base for consistency
    return script_dir

def process_uploaded_csv(unique_id):
    """Process CSV file from upload directory with unique ID - works from any path"""
    base_path = get_base_path()
    
    # Define upload path structure relative to script location
    upload_path = base_path / "uploads" / unique_id / "input.csv"
    output_dir = base_path / "outputs" / unique_id
    output_file = output_dir / "analysis.json"
    
    # Check if input file exists - return specific message for frontend
    if not upload_path.exists():
        error_result = {
            "error": "No file uploaded from frontend", 
            "status": "no_file",
            "expected_path": str(upload_path),
            "unique_id": unique_id
        }
        return error_result, None
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and process CSV
    print(f"Processing file: {upload_path}")
    df = read_csv_file(upload_path)
    if df is None:
        return {"error": "Failed to read CSV file", "file_path": str(upload_path)}, None
    
    # Analyze data
    analysis_result = analyze_csv_data(df, "input.csv")
    if analysis_result is None:
        return {"error": "Analysis failed"}, None
    
    # Get data type tuples
    numerical_cols, categorical_cols, datetime_cols = get_data_type_separation(df)
    
    # Add processing info to analysis
    analysis_result["processing_info"] = {
        "unique_id": unique_id,
        "input_path": str(upload_path),
        "output_path": str(output_file),
        "processing_status": "completed",
        "base_path": str(base_path)
    }
    
    # Save JSON output
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON Analysis saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving JSON: {str(e)}")
        return {"error": f"Failed to save JSON: {str(e)}"}, None
    
    print(f"✅ Analysis completed for ID: {unique_id}")
    
    return analysis_result, (numerical_cols, categorical_cols, datetime_cols)

def watch_for_uploads(poll_interval=2):
    """Automatically watch for file uploads and process them"""
    base_path = get_base_path()
    uploads_dir = base_path / "uploads"
    outputs_dir = base_path / "outputs"
    
    # Create directories if they don't exist
    uploads_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Watching for uploads in: {uploads_dir}")
    print(f"📁 Outputs will be saved to: {outputs_dir}")
    print("Press Ctrl+C to stop watching...")
    
    processed_files = set()
    
    try:
        while True:
            # Look for new upload directories
            for unique_dir in uploads_dir.glob("*"):
                if unique_dir.is_dir():
                    unique_id = unique_dir.name
                    input_file = unique_dir / "input.csv"
                    output_file = outputs_dir / unique_id / "analysis.json"
                    
                    # Skip if already processed
                    file_key = f"{unique_id}_{input_file.stat().st_mtime if input_file.exists() else 0}"
                    if file_key in processed_files:
                        continue
                    
                    # Check if input file exists
                    if input_file.exists():
                        print(f"\n🚀 New file detected: {unique_id}/input.csv")
                        
                        # Process the file
                        result, tuples = process_uploaded_csv(unique_id)
                        
                        if "error" not in result:
                            print(f"✅ Successfully processed: {unique_id}")
                            print(f"📊 Records: {result['summary_stats']['total_records']}")
                            print(f"📊 Variables: {result['summary_stats']['total_variables']}")
                            print(f"📊 Data Types: {tuples}")
                        else:
                            print(f"❌ Processing failed: {result['error']}")
                        
                        processed_files.add(file_key)
            
            time.sleep(poll_interval)
    
    except KeyboardInterrupt:
        print("\n👋 Stopped watching for uploads")

def test_function():
    """Test function - creates sample data and tests everything"""
    print("🧪 TESTING: Creating sample data and testing all functions...")
    
    base_path = get_base_path()
    print(f"📁 Working from base path: {base_path}")
    
    # Create sample test data
    sample_data = {
        'customerID': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No', 'No', 'Yes'],
        'tenure': [12, 24, 6, 48, 36],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'InternetService': ['DSL', 'Fiber', 'No', 'DSL', 'Fiber'],
        'MonthlyCharges': [29.85, 56.95, 20.05, 42.30, 78.70],
        'TotalCharges': ['358.2', '1367.8', '120.3', '2031.4', '2835.2'],
        'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
    }
    
    # Create test directories
    test_dir = base_path / "test_sample"
    test_upload_dir = base_path / "uploads" / "test123"
    test_upload_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Save sample data as CSV
    df = pd.DataFrame(sample_data)
    test_file = test_upload_dir / "input.csv"
    df.to_csv(test_file, index=False)
    
    print(f"✅ Created test file: {test_file}")
    print(f"📊 Test data: {len(df)} rows, {len(df.columns)} columns")
    
    # Test the analysis functions
    print("\n🔍 Testing analysis functions...")
    
    # Test data type separation
    numerical_cols, categorical_cols, datetime_cols = get_data_type_separation(df)
    print(f"✅ Data type separation:")
    print(f"   Numerical: {numerical_cols}")
    print(f"   Categorical: {categorical_cols}")
    print(f"   DateTime: {datetime_cols}")
    
    # Test full processing
    print(f"\n🚀 Testing full upload processing...")
    result, tuples = process_uploaded_csv("test123")
    
    if "error" not in result:
        print(f"✅ Processing successful!")
        print(f"📊 JSON Output Preview:")
        print(f"   Total records: {result['summary_stats']['total_records']}")
        print(f"   Total variables: {result['summary_stats']['total_variables']}")
        print(f"   Completeness: {result['summary_stats']['completeness_percentage']}%")
        print(f"📊 Tuples returned: {tuples}")
        
        # Show JSON file location
        json_file = base_path / "outputs" / "test123" / "analysis.json"
        if json_file.exists():
            print(f"✅ JSON saved successfully to: {json_file}")
            
            # Show actual JSON content
            with open(json_file, 'r') as f:
                json_content = json.load(f)
            print(f"\n📄 JSON Content Structure:")
            for key in json_content.keys():
                print(f"   - {key}")
        else:
            print(f"❌ JSON file not found at: {json_file}")
    else:
        print(f"❌ Processing failed: {result['error']}")
    
    # Test no file case
    print(f"\n🚫 Testing 'no file' scenario...")
    no_file_result, _ = process_uploaded_csv("nonexistent_id")
    if "No file uploaded from frontend" in no_file_result.get("error", ""):
        print("✅ 'No file uploaded' message working correctly")
    
    print(f"\n🎉 All tests completed!")
    return result, tuples

# Main functions for external use
def main_process_upload(unique_id):
    """Main function to process uploaded file with unique ID"""
    return process_uploaded_csv(unique_id)

def main_analyze_file(file_path, output_path=None):
    """Main function to analyze any CSV file directly"""
    df = read_csv_file(file_path)
    if df is None:
        return None, None
    
    file_name = os.path.basename(file_path)
    analysis = analyze_csv_data(df, file_name)
    data_types = get_data_type_separation(df)
    
    if analysis and output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"✅ Analysis saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving JSON: {str(e)}")
    
    return analysis, data_types

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py upload <unique_id>     # Process specific uploaded file")
        print("  python script.py watch                  # Auto-watch for uploads")
        print("  python script.py test                   # Run test with sample data")
        print("  python script.py analyze <file_path>    # Analyze specific file")
    
    elif sys.argv[1] == "upload" and len(sys.argv) > 2:
        unique_id = sys.argv[2]
        result, data_types = main_process_upload(unique_id)
        
        if "error" not in result:
            print(f"✅ Processing completed for ID: {unique_id}")
            print(f"📊 Data Types: {data_types}")
            print(f"📄 JSON saved to outputs/{unique_id}/analysis.json")
        else:
            print(f"❌ {result['error']}")
    
    elif sys.argv[1] == "watch":
        watch_for_uploads()
    
    elif sys.argv[1] == "test":
        print("🧪 Running test function...")
        analysis, tuples = test_function()
        if analysis and "error" not in analysis:
            print(f"\n✅ Test completed successfully!")
        else:
            print(f"\n❌ Test failed!")
    
    elif sys.argv[1] == "analyze" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        
        analysis, data_types = main_analyze_file(file_path, output_path)
        
        if analysis:
            print(f"✅ Analysis completed")
            print(f"📊 Records: {analysis['summary_stats']['total_records']}")
            print(f"📊 Variables: {analysis['summary_stats']['total_variables']}")
            print(f"📊 Data Types: {data_types}")
        else:
            print("❌ Analysis failed")
    
    else:
        print("Invalid arguments. Use 'upload <id>', 'watch', 'test', or 'analyze <file>'")