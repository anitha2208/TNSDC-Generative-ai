import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uuid
from typing import Dict, Any
import io

app = FastAPI(title="CSV Data Analysis API", version="1.0.0")

class CSVAnalyzer:
    def __init__(self):
        self.df = None
        self.file_name = None
        self.analysis_results = {}
        
    def read_csv_from_upload(self, file_content: bytes, filename: str):
        """Read CSV from uploaded file content"""
        self.file_name = filename
        
        try:
            # Convert bytes to string IO for pandas
            csv_string = file_content.decode('utf-8')
            self.df = pd.read_csv(io.StringIO(csv_string))
            return True
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
    
    def analyze_dataset(self):
        """Comprehensive data analysis"""
        if self.df is None:
            return None
        
        analysis = {}
        
        # Basic dataset information
        analysis['basic_info'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'file_name': self.file_name,
            'columns': list(self.df.columns)
        }
        
        # Data types analysis
        analysis['data_types'] = {
            'numerical_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
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
            analysis['numerical_summary'] = {
                col: {stat: float(val) for stat, val in col_stats.items()} 
                for col, col_stats in numerical_summary.to_dict().items()
            }
        
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
    
    def generate_left_division_data(self):
        """Generate structured data for left division"""
        if not self.analysis_results:
            return {}
        
        analysis = self.analysis_results
        
        left_division = {
            "dataset_overview": {
                "total_records": analysis['basic_info']['total_rows'],
                "feature_variables": analysis['basic_info']['total_columns'],
                "numerical_fields": len(analysis['data_types']['numerical_columns']),
                "categorical_fields": len(analysis['data_types']['categorical_columns']),
                "datetime_fields": len(analysis['data_types']['datetime_columns'])
            },
            "data_quality_assessment": {
                "overall_completeness": round(analysis['data_quality']['completeness_score'], 1),
                "duplicate_records": analysis['data_quality']['duplicate_rows'],
                "quality_rating": self._get_quality_rating(analysis['data_quality']['completeness_score'])
            },
            "missing_data_impact": {
                "has_missing_data": analysis['missing_values']['total_missing'] > 0,
                "affected_columns": len(analysis['missing_values']['columns_with_missing']),
                "total_missing_values": analysis['missing_values']['total_missing'],
                "top_missing_columns": self._get_top_missing_columns(analysis)
            }
        }
        
        return left_division
    
    def generate_right_division_data(self):
        """Generate structured data for right division"""
        if not self.analysis_results:
            return {}
        
        analysis = self.analysis_results
        
        right_division = {
            "analytical_capabilities": {
                "statistical_analysis_ready": len(analysis['data_types']['numerical_columns']) > 0,
                "segmentation_analysis_available": len(analysis['data_types']['categorical_columns']) > 0,
                "mixed_type_analysis_supported": (len(analysis['data_types']['numerical_columns']) > 0 and 
                                               len(analysis['data_types']['categorical_columns']) > 0),
                "available_analyses": self._get_available_analyses(analysis)
            },
            "business_value_indicators": {
                "data_scale": self._assess_data_scale(analysis['basic_info']['total_rows']),
                "variable_richness": self._assess_variable_richness(analysis['basic_info']['total_columns']),
                "reliability_level": self._assess_reliability(analysis['data_quality']['completeness_score'])
            },
            "key_applications": self._determine_applications(analysis)
        }
        
        return right_division
    
    def _get_quality_rating(self, completeness_score):
        """Get quality rating based on completeness score"""
        if completeness_score > 95:
            return "Excellent"
        elif completeness_score > 85:
            return "Good"
        elif completeness_score > 70:
            return "Fair"
        else:
            return "Poor"
    
    def _get_top_missing_columns(self, analysis):
        """Get top columns with missing data"""
        if not analysis['missing_values']['columns_with_missing']:
            return []
        
        missing_cols = analysis['missing_values']['columns_with_missing']
        missing_percentages = analysis['missing_values']['missing_percentages']
        
        sorted_missing = sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return [
            {
                "column": col,
                "missing_count": missing_count,
                "missing_percentage": round(missing_percentages.get(col, 0), 1)
            }
            for col, missing_count in sorted_missing
        ]
    
    def _get_available_analyses(self, analysis):
        """Get list of available analyses"""
        analyses = []
        
        numerical_cols = len(analysis['data_types']['numerical_columns'])
        categorical_cols = len(analysis['data_types']['categorical_columns'])
        
        if numerical_cols > 0:
            analyses.extend([
                "statistical_analysis",
                "trend_analysis", 
                "correlation_studies",
                "predictive_modeling"
            ])
        
        if categorical_cols > 0:
            analyses.extend([
                "segmentation_analysis",
                "classification_studies"
            ])
        
        if numerical_cols > 0 and categorical_cols > 0:
            analyses.append("mixed_type_analysis")
        
        return analyses
    
    def _assess_data_scale(self, row_count):
        """Assess data scale based on row count"""
        if row_count > 10000:
            return {
                "scale": "large",
                "description": "Large-scale dataset for enterprise insights"
            }
        elif row_count > 1000:
            return {
                "scale": "medium",
                "description": "Medium-scale dataset for departmental analysis"
            }
        else:
            return {
                "scale": "focused",
                "description": "Focused dataset for targeted analysis"
            }
    
    def _assess_variable_richness(self, var_count):
        """Assess variable richness based on column count"""
        if var_count > 20:
            return {
                "richness": "rich",
                "description": "Rich feature set for comprehensive analysis"
            }
        elif var_count > 10:
            return {
                "richness": "adequate",
                "description": "Adequate variables for detailed insights"
            }
        else:
            return {
                "richness": "focused",
                "description": "Focused variable set for specific analysis"
            }
    
    def _assess_reliability(self, completeness_score):
        """Assess reliability based on completeness score"""
        if completeness_score > 90:
            return {
                "level": "high",
                "description": "High reliability for business decisions"
            }
        elif completeness_score > 75:
            return {
                "level": "moderate", 
                "description": "Moderate reliability with some limitations"
            }
        else:
            return {
                "level": "low",
                "description": "Requires data cleaning for optimal use"
            }
    
    def _determine_applications(self, analysis):
        """Determine key applications based on data characteristics"""
        file_name = analysis['basic_info']['file_name'].lower()
        columns = [col.lower() for col in analysis['basic_info']['columns']]
        
        applications = []
        
        if any(keyword in file_name or any(keyword in col for col in columns) 
               for keyword in ['sales', 'revenue', 'price', 'cost']):
            applications = [
                "sales_performance_analysis",
                "revenue_optimization_studies", 
                "pricing_strategy_development"
            ]
        elif any(keyword in file_name or any(keyword in col for col in columns) 
                 for keyword in ['customer', 'client', 'user']):
            applications = [
                "customer_behavior_analysis",
                "market_segmentation_studies",
                "customer_lifetime_value_modeling"
            ]
        elif any(keyword in file_name or any(keyword in col for col in columns) 
                 for keyword in ['employee', 'staff', 'hr', 'payroll']):
            applications = [
                "workforce_analytics",
                "performance_management_insights",
                "hr_optimization_strategies"
            ]
        else:
            applications = [
                "exploratory_data_analysis",
                "pattern_recognition_studies", 
                "statistical_modeling_projects",
                "business_intelligence_reporting"
            ]
        
        return applications
    
    def generate_complete_analysis(self):
        """Generate complete analysis in JSON format"""
        if not self.analysis_results:
            return {}
        
        analysis = self.analysis_results
        completeness = analysis['data_quality']['completeness_score']
        
        complete_analysis = {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "file_info": {
                "filename": analysis['basic_info']['file_name'],
                "total_records": analysis['basic_info']['total_rows'],
                "total_variables": analysis['basic_info']['total_columns'],
                "completeness_percentage": round(completeness, 1)
            },
            "summary_stats": f"{analysis['basic_info']['total_rows']:,} Records • {analysis['basic_info']['total_columns']} Variables • {completeness:.1f}% Complete",
            "divisions": {
                "left_division": self.generate_left_division_data(),
                "right_division": self.generate_right_division_data()
            },
            "raw_analysis": analysis
        }
        
        return complete_analysis

# Initialize the analyzer
analyzer = CSVAnalyzer()

@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Analyze uploaded CSV file and return structured JSON analysis
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process the file
        analyzer.read_csv_from_upload(file_content, file.filename)
        analyzer.analyze_dataset()
        
        # Generate complete analysis
        analysis_result = analyzer.generate_complete_analysis()
        
        return JSONResponse(content=analysis_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "CSV Analysis API", "version": "1.0.0", "endpoints": ["/analyze-csv"]}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# For testing purposes, you can also add a direct function call
def analyze_csv_file(file_path: str) -> Dict[str, Any]:
    """
    Direct function to analyze CSV file from file path
    """
    analyzer_instance = CSVAnalyzer()
    
    try:
        # Read file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        filename = os.path.basename(file_path)
        analyzer_instance.read_csv_from_upload(file_content, filename)
        analyzer_instance.analyze_dataset()
        
        return analyzer_instance.generate_complete_analysis()
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

# Example usage function
def main():
    """Example of how to use the analyzer directly"""
    file_path = input("Enter CSV file path: ").strip().strip('"')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    result = analyze_csv_file(file_path)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)