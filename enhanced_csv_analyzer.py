import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uuid
from typing import Dict, Any, List, Optional
import io
from pathlib import Path

app = FastAPI(title="Enhanced CSV Data Analysis API", version="2.0.0")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class EnhancedCSVAnalyzer:
    def __init__(self):
        self.df = None
        self.file_name = None
        self.analysis_id = None
        self.input_file_path = None
        self.analysis_results = {}
        self.predefined_rules = self._initialize_predefined_rules()
        
    def _initialize_predefined_rules(self) -> Dict[str, Any]:
        """Initialize comprehensive predefined rules for data analysis"""
        return {
            "numerical_data_rules": {
                "statistical_thresholds": {
                    "outlier_z_score": 3.0,
                    "skewness_normal_range": [-0.5, 0.5],
                    "kurtosis_normal_range": [-2.0, 2.0],
                    "correlation_strong_threshold": 0.8,
                    "correlation_moderate_threshold": 0.5
                },
                "data_quality_rules": {
                    "excellent_completeness": 0.95,
                    "good_completeness": 0.85,
                    "acceptable_completeness": 0.70,
                    "max_zero_percentage": 0.3,
                    "min_variance_threshold": 0.01,
                    "infinite_values_threshold": 0.01
                },
                "business_classification": {
                    "financial_keywords": ["revenue", "sales", "income", "profit", "cost", "expense", "price", "amount", "total", "balance"],
                    "performance_keywords": ["score", "rating", "performance", "efficiency", "productivity", "kpi"],
                    "quantity_keywords": ["quantity", "count", "volume", "number", "size", "length", "weight", "duration"],
                    "percentage_keywords": ["percentage", "percent", "rate", "ratio", "proportion"]
                },
                "transformation_rules": {
                    "log_transform_skewness": 2.0,
                    "standardization_different_scales": True,
                    "normalization_range": [0, 1]
                }
            },
            "categorical_data_rules": {
                "cardinality_classification": {
                    "binary": 2,
                    "low_cardinality": 10,
                    "medium_cardinality": 50,
                    "high_cardinality": 100,
                    "very_high_cardinality": 500
                },
                "encoding_strategies": {
                    "one_hot_max_categories": 10,
                    "label_encoding_max_categories": 50,
                    "target_encoding_min_categories": 20,
                    "frequency_encoding_threshold": 100
                },
                "data_quality_rules": {
                    "excellent_completeness": 0.98,
                    "good_completeness": 0.90,
                    "acceptable_completeness": 0.80,
                    "max_inconsistency_threshold": 0.05,
                    "min_category_frequency": 0.01
                },
                "business_classification": {
                    "identifier_keywords": ["id", "code", "key", "identifier", "reference"],
                    "demographic_keywords": ["gender", "age", "education", "income", "status", "level"],
                    "geographic_keywords": ["country", "state", "city", "region", "location", "address", "zip"],
                    "temporal_keywords": ["month", "quarter", "season", "period", "year", "day"],
                    "product_keywords": ["category", "type", "brand", "model", "variant", "class"],
                    "behavioral_keywords": ["preference", "choice", "behavior", "pattern", "segment"]
                }
            },
            "data_validation_rules": {
                "dataset_quality": {
                    "minimum_rows": 100,
                    "recommended_rows": 1000,
                    "minimum_columns": 2,
                    "max_duplicate_percentage": 0.10,
                    "overall_completeness_threshold": 0.85
                },
                "analysis_readiness": {
                    "min_numerical_columns": 1,
                    "min_categorical_columns": 1,
                    "mixed_data_bonus": True
                }
            },
            "preprocessing_recommendations": {
                "missing_data_strategies": {
                    "numerical": {
                        "low_missing": "mean_imputation",
                        "medium_missing": "median_imputation", 
                        "high_missing": "predictive_imputation"
                    },
                    "categorical": {
                        "low_missing": "mode_imputation",
                        "medium_missing": "frequent_category_imputation",
                        "high_missing": "create_missing_category"
                    }
                },
                "outlier_handling": {
                    "mild_outliers": "cap_at_percentiles",
                    "extreme_outliers": "remove_or_transform",
                    "domain_specific": "business_rule_validation"
                }
            }
        }
    
    def generate_unique_analysis_id(self) -> str:
        """Generate unique analysis ID"""
        self.analysis_id = str(uuid.uuid4())
        return self.analysis_id
    
    def save_uploaded_file_as_input_csv(self, file_content: bytes, original_filename: str) -> str:
        """Save uploaded file as input.csv with unique ID prefix"""
        self.analysis_id = self.generate_unique_analysis_id()
        
        # Always save as input.csv with analysis ID prefix
        input_filename = f"{self.analysis_id}_input.csv"
        self.input_file_path = UPLOAD_DIR / input_filename
        self.file_name = original_filename
        
        with open(self.input_file_path, 'wb') as f:
            f.write(file_content)
        
        return str(self.input_file_path)
    
    def load_csv_from_input_file(self):
        """Load CSV data from the saved input.csv file"""
        if not self.input_file_path or not os.path.exists(self.input_file_path):
            raise HTTPException(status_code=404, detail="Input CSV file not found")
        
        try:
            self.df = pd.read_csv(self.input_file_path)
            return True
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading input CSV file: {str(e)}")
    
    def separate_numerical_categorical_data(self) -> Dict[str, Any]:
        """Separate data into numerical and categorical with enhanced logic"""
        if self.df is None:
            return {}
        
        # Initial separation based on data types
        numerical_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = list(self.df.select_dtypes(include=['datetime64']).columns)
        
        # Enhanced logic to reclassify columns based on content
        refined_numerical = []
        refined_categorical = []
        
        # Check numerical columns for categorical nature
        for col in numerical_cols:
            unique_values = self.df[col].nunique()
            total_values = len(self.df[col].dropna())
            
            # If numerical column has very few unique values, it might be categorical
            if unique_values <= 10 and total_values > unique_values * 10:
                refined_categorical.append(col)
            else:
                refined_numerical.append(col)
        
        # Check categorical columns for numerical nature
        for col in categorical_cols:
            # Try to convert to numeric
            try:
                pd.to_numeric(self.df[col], errors='raise')
                refined_numerical.append(col)
            except:
                refined_categorical.append(col)
        
        separation_results = {
            "numerical_columns": refined_numerical,
            "categorical_columns": refined_categorical,
            "datetime_columns": datetime_cols,
            "total_numerical": len(refined_numerical),
            "total_categorical": len(refined_categorical),
            "total_datetime": len(datetime_cols),
            "separation_summary": {
                "data_type_distribution": {
                    "numerical_percentage": (len(refined_numerical) / len(self.df.columns)) * 100,
                    "categorical_percentage": (len(refined_categorical) / len(self.df.columns)) * 100,
                    "datetime_percentage": (len(datetime_cols) / len(self.df.columns)) * 100
                }
            }
        }
        
        return separation_results
    
    def apply_predefined_rules_to_numerical_data(self, numerical_columns: List[str]) -> Dict[str, Any]:
        """Apply comprehensive predefined rules to numerical data"""
        if not numerical_columns:
            return {"message": "No numerical columns available for analysis"}
        
        rules = self.predefined_rules["numerical_data_rules"]
        numerical_df = self.df[numerical_columns]
        
        results = {
            "columns_analyzed": numerical_columns,
            "column_count": len(numerical_columns),
            "detailed_analysis": {},
            "summary_statistics": {},
            "data_quality_assessment": {},
            "business_classification": {},
            "transformation_recommendations": {},
            "preprocessing_suggestions": [],
            "correlation_analysis": {},
            "outlier_analysis": {}
        }
        
        # Detailed analysis for each column
        for col in numerical_columns:
            col_data = numerical_df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Basic statistics
            basic_stats = {
                "count": int(len(col_data)),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q25": float(col_data.quantile(0.25)),
                "q75": float(col_data.quantile(0.75)),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis())
            }
            
            # Advanced statistics
            advanced_stats = {
                "range": float(col_data.max() - col_data.min()),
                "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                "cv": float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0,
                "unique_values": int(col_data.nunique()),
                "zero_count": int((col_data == 0).sum()),
                "zero_percentage": float((col_data == 0).sum() / len(col_data) * 100)
            }
            
            # Data quality assessment
            completeness = len(col_data) / len(numerical_df)
            quality_score = self._calculate_numerical_quality_score(col_data, rules)
            
            quality_assessment = {
                "completeness": float(completeness),
                "quality_score": quality_score,
                "quality_grade": self._get_quality_grade(quality_score),
                "has_outliers": self._detect_outliers(col_data, rules),
                "distribution_shape": self._assess_distribution_shape(basic_stats["skewness"], basic_stats["kurtosis"], rules)
            }
            
            # Business classification
            business_type = self._classify_numerical_business_type(col, rules)
            
            # Transformation recommendations
            transformation_recs = self._get_transformation_recommendations(col_data, basic_stats, rules)
            
            # Store results
            results["detailed_analysis"][col] = {
                "basic_statistics": basic_stats,
                "advanced_statistics": advanced_stats
            }
            results["data_quality_assessment"][col] = quality_assessment
            results["business_classification"][col] = business_type
            results["transformation_recommendations"][col] = transformation_recs
        
        # Correlation analysis
        if len(numerical_columns) > 1:
            correlation_matrix = numerical_df.corr()
            results["correlation_analysis"] = self._analyze_correlations(correlation_matrix, rules)
        
        # Overall outlier analysis
        results["outlier_analysis"] = self._comprehensive_outlier_analysis(numerical_df, rules)
        
        # Generate preprocessing suggestions
        results["preprocessing_suggestions"] = self._generate_numerical_preprocessing_suggestions(results, rules)
        
        return results
    
    def apply_predefined_rules_to_categorical_data(self, categorical_columns: List[str]) -> Dict[str, Any]:
        """Apply comprehensive predefined rules to categorical data"""
        if not categorical_columns:
            return {"message": "No categorical columns available for analysis"}
        
        rules = self.predefined_rules["categorical_data_rules"]
        categorical_df = self.df[categorical_columns]
        
        results = {
            "columns_analyzed": categorical_columns,
            "column_count": len(categorical_columns),
            "detailed_analysis": {},
            "cardinality_analysis": {},
            "data_quality_assessment": {},
            "business_classification": {},
            "encoding_recommendations": {},
            "preprocessing_suggestions": [],
            "frequency_analysis": {}
        }
        
        # Detailed analysis for each column
        for col in categorical_columns:
            col_data = categorical_df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Basic categorical statistics
            unique_count = col_data.nunique()
            value_counts = col_data.value_counts()
            
            basic_stats = {
                "total_values": int(len(col_data)),
                "unique_values": int(unique_count),
                "most_frequent_value": str(value_counts.index[0]) if len(value_counts) > 0 else "N/A",
                "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "least_frequent_value": str(value_counts.index[-1]) if len(value_counts) > 0 else "N/A",
                "least_frequent_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
            }
            
            # Cardinality analysis
            cardinality_info = self._analyze_cardinality(unique_count, len(col_data), rules)
            
            # Data quality assessment
            completeness = len(col_data) / len(categorical_df)
            quality_assessment = {
                "completeness": float(completeness),
                "consistency_score": self._calculate_categorical_consistency(col_data),
                "quality_score": self._calculate_categorical_quality_score(col_data, completeness, rules),
                "has_rare_categories": self._detect_rare_categories(value_counts, rules),
                "distribution_balance": self._assess_category_balance(value_counts)
            }
            
            # Business classification
            business_type = self._classify_categorical_business_type(col, rules)
            
            # Encoding recommendations
            encoding_recs = self._get_encoding_recommendations(unique_count, value_counts, rules)
            
            # Frequency analysis
            frequency_analysis = {
                "top_10_categories": {str(k): int(v) for k, v in value_counts.head(10).items()},
                "category_distribution": self._analyze_category_distribution(value_counts),
                "entropy": self._calculate_entropy(value_counts)
            }
            
            # Store results
            results["detailed_analysis"][col] = basic_stats
            results["cardinality_analysis"][col] = cardinality_info
            results["data_quality_assessment"][col] = quality_assessment
            results["business_classification"][col] = business_type
            results["encoding_recommendations"][col] = encoding_recs
            results["frequency_analysis"][col] = frequency_analysis
        
        # Generate preprocessing suggestions
        results["preprocessing_suggestions"] = self._generate_categorical_preprocessing_suggestions(results, rules)
        
        return results
    
    def _calculate_numerical_quality_score(self, data: pd.Series, rules: Dict) -> float:
        """Calculate comprehensive quality score for numerical data"""
        score = 0.0
        
        # Completeness score (25%)
        completeness = len(data) / len(self.df)
        if completeness >= rules["data_quality_rules"]["excellent_completeness"]:
            score += 0.25
        elif completeness >= rules["data_quality_rules"]["good_completeness"]:
            score += 0.20
        elif completeness >= rules["data_quality_rules"]["acceptable_completeness"]:
            score += 0.15
        else:
            score += 0.25 * completeness
        
        # Variance score (20%)
        if data.var() >= rules["data_quality_rules"]["min_variance_threshold"]:
            score += 0.20
        else:
            score += 0.20 * (data.var() / rules["data_quality_rules"]["min_variance_threshold"])
        
        # Zero percentage score (20%)
        zero_pct = (data == 0).sum() / len(data)
        if zero_pct <= rules["data_quality_rules"]["max_zero_percentage"]:
            score += 0.20
        else:
            score += 0.20 * (1 - zero_pct)
        
        # Outlier score (20%)
        z_scores = np.abs((data - data.mean()) / data.std())
        outlier_pct = (z_scores > rules["statistical_thresholds"]["outlier_z_score"]).sum() / len(data)
        if outlier_pct <= 0.05:  # 5% outliers acceptable
            score += 0.20
        else:
            score += 0.20 * (1 - outlier_pct)
        
        # Distribution normality score (15%)
        skewness = abs(data.skew())
        if rules["statistical_thresholds"]["skewness_normal_range"][0] <= skewness <= rules["statistical_thresholds"]["skewness_normal_range"][1]:
            score += 0.15
        else:
            score += 0.15 * max(0, 1 - skewness / 3)
        
        return min(score, 1.0)
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        else:
            return "Poor"
    
    def _detect_outliers(self, data: pd.Series, rules: Dict) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        z_scores = np.abs((data - data.mean()) / data.std())
        z_outliers = z_scores > rules["statistical_thresholds"]["outlier_z_score"]
        
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = (data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)
        
        return {
            "has_z_score_outliers": bool(z_outliers.any()),
            "z_score_outlier_count": int(z_outliers.sum()),
            "z_score_outlier_percentage": float(z_outliers.sum() / len(data) * 100),
            "has_iqr_outliers": bool(iqr_outliers.any()),
            "iqr_outlier_count": int(iqr_outliers.sum()),
            "iqr_outlier_percentage": float(iqr_outliers.sum() / len(data) * 100)
        }
    
    def _assess_distribution_shape(self, skewness: float, kurtosis: float, rules: Dict) -> Dict[str, Any]:
        """Assess the shape of the distribution"""
        skew_range = rules["statistical_thresholds"]["skewness_normal_range"]
        kurt_range = rules["statistical_thresholds"]["kurtosis_normal_range"]
        
        if skew_range[0] <= skewness <= skew_range[1]:
            skew_assessment = "normal"
        elif skewness > skew_range[1]:
            skew_assessment = "right_skewed"
        else:
            skew_assessment = "left_skewed"
        
        if kurt_range[0] <= kurtosis <= kurt_range[1]:
            kurt_assessment = "mesokurtic"
        elif kurtosis > kurt_range[1]:
            kurt_assessment = "leptokurtic"
        else:
            kurt_assessment = "platykurtic"
        
        return {
            "skewness_assessment": skew_assessment,
            "kurtosis_assessment": kurt_assessment,
            "overall_shape": f"{skew_assessment}_{kurt_assessment}",
            "normality_score": self._calculate_normality_score(skewness, kurtosis, rules)
        }
    
    def _calculate_normality_score(self, skewness: float, kurtosis: float, rules: Dict) -> float:
        """Calculate normality score based on skewness and kurtosis"""
        skew_range = rules["statistical_thresholds"]["skewness_normal_range"]
        kurt_range = rules["statistical_thresholds"]["kurtosis_normal_range"]
        
        skew_score = 1.0 if skew_range[0] <= skewness <= skew_range[1] else max(0, 1 - abs(skewness) / 3)
        kurt_score = 1.0 if kurt_range[0] <= kurtosis <= kurt_range[1] else max(0, 1 - abs(kurtosis) / 5)
        
        return (skew_score + kurt_score) / 2
    
    def _classify_numerical_business_type(self, column_name: str, rules: Dict) -> Dict[str, Any]:
        """Classify numerical column based on business context"""
        col_lower = column_name.lower()
        classification = "general_numerical"
        confidence = 0.0
        
        for category, keywords in rules["business_classification"].items():
            matches = sum(1 for keyword in keywords if keyword in col_lower)
            if matches > 0:
                current_confidence = matches / len(keywords)
                if current_confidence > confidence:
                    classification = category.replace("_keywords", "")
                    confidence = current_confidence
        
        return {
            "business_type": classification,
            "confidence_score": float(confidence),
            "suggested_use_cases": self._get_numerical_use_cases(classification)
        }
    
    def _get_numerical_use_cases(self, business_type: str) -> List[str]:
        """Get suggested use cases based on business type"""
        use_cases_map = {
            "financial": ["Revenue Analysis", "Cost Optimization", "Profitability Studies", "Financial Forecasting"],
            "performance": ["KPI Tracking", "Performance Benchmarking", "Efficiency Analysis", "Quality Metrics"],
            "quantity": ["Inventory Management", "Volume Analysis", "Capacity Planning", "Resource Allocation"],
            "percentage": ["Rate Analysis", "Proportion Studies", "Conversion Tracking", "Success Rate Monitoring"],
            "general_numerical": ["Statistical Analysis", "Trend Analysis", "Correlation Studies", "Predictive Modeling"]
        }
        return use_cases_map.get(business_type, use_cases_map["general_numerical"])
    
    def _get_transformation_recommendations(self, data: pd.Series, stats: Dict, rules: Dict) -> Dict[str, Any]:
        """Get transformation recommendations for numerical data"""
        recommendations = []
        
        # Skewness-based recommendations
        if abs(stats["skewness"]) > rules["transformation_rules"]["log_transform_skewness"]:
            if data.min() > 0:
                recommendations.append({
                    "transformation": "log_transform",
                    "reason": f"High skewness ({stats['skewness']:.2f}) and positive values",
                    "priority": "high"
                })
            else:
                recommendations.append({
                    "transformation": "box_cox_transform", 
                    "reason": f"High skewness ({stats['skewness']:.2f}) with zero/negative values",
                    "priority": "medium"
                })
        
        # Scaling recommendations
        if stats["std"] > 1000 or stats["max"] - stats["min"] > 1000:
            recommendations.append({
                "transformation": "standardization",
                "reason": "Large scale differences detected",
                "priority": "medium"
            })
        
        # Outlier handling
        z_scores = np.abs((data - data.mean()) / data.std())
        if (z_scores > rules["statistical_thresholds"]["outlier_z_score"]).any():
            recommendations.append({
                "transformation": "outlier_treatment",
                "reason": "Outliers detected that may skew analysis",
                "priority": "high"
            })
        
        return {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "high_priority_count": sum(1 for r in recommendations if r["priority"] == "high")
        }
    
    def _analyze_correlations(self, corr_matrix: pd.DataFrame, rules: Dict) -> Dict[str, Any]:
        """Analyze correlations between numerical variables"""
        strong_threshold = rules["statistical_thresholds"]["correlation_strong_threshold"]
        moderate_threshold = rules["statistical_thresholds"]["correlation_moderate_threshold"]
        
        # Find strong correlations
        strong_correlations = []
        moderate_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]
                
                if abs(corr_value) >= strong_threshold:
                    strong_correlations.append({
                        "variable_1": var1,
                        "variable_2": var2, 
                        "correlation": float(corr_value),
                        "strength": "strong"
                    })
                elif abs(corr_value) >= moderate_threshold:
                    moderate_correlations.append({
                        "variable_1": var1,
                        "variable_2": var2,
                        "correlation": float(corr_value),
                        "strength": "moderate"
                    })
        
        return {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "strong_correlations": strong_correlations,
            "moderate_correlations": moderate_correlations,
            "total_strong_correlations": len(strong_correlations),
            "total_moderate_correlations": len(moderate_correlations),
            "multicollinearity_risk": len(strong_correlations) > 0
        }
    
    def _comprehensive_outlier_analysis(self, numerical_df: pd.DataFrame, rules: Dict) -> Dict[str, Any]:
        """Comprehensive outlier analysis across all numerical columns"""
        outlier_summary = {
            "columns_with_outliers": [],
            "total_outlier_records": 0,
            "outlier_percentage": 0.0,
            "outlier_treatment_priority": []
        }
        
        outlier_records = set()
        
        for col in numerical_df.columns:
            col_data = numerical_df[col].dropna()
            if len(col_data) == 0:
                continue
                
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            col_outliers = z_scores > rules["statistical_thresholds"]["outlier_z_score"]
            
            if col_outliers.any():
                outlier_count = col_outliers.sum()
                outlier_pct = outlier_count / len(col_data) * 100
                
                outlier_summary["columns_with_outliers"].append({
                    "column": col,
                    "outlier_count": int(outlier_count),
                    "outlier_percentage": float(outlier_pct),
                    "severity": "high" if outlier_pct > 10 else "medium" if outlier_pct > 5 else "low"
                })
                
                # Track record indices with outliers
                outlier_indices = col_data[col_outliers].index
                outlier_records.update(outlier_indices)
        
        outlier_summary["total_outlier_records"] = len(outlier_records)
        outlier_summary["outlier_percentage"] = len(outlier_records) / len(numerical_df) * 100
        
        return outlier_summary
    
    def _generate_numerical_preprocessing_suggestions(self, results: Dict, rules: Dict) -> List[str]:
        """Generate preprocessing suggestions for numerical data"""
        suggestions = []
        
        # Quality-based suggestions
        for col, quality in results["data_quality_assessment"].items():
            if quality["quality_score"] < 0.7:
                suggestions.append(f"Column '{col}' has quality issues (score: {quality['quality_score']:.2f}). Consider data cleaning.")
            
            if quality["has_outliers"]["iqr_outlier_percentage"] > 10:
                suggestions.append(f"Column '{col}' has significant outliers ({quality['has_outliers']['iqr_outlier_percentage']:.1f}%). Consider outlier treatment.")
        
        # Correlation-based suggestions
        if "correlation_analysis" in results and results["correlation_analysis"]["multicollinearity_risk"]:
            suggestions.append("Strong correlations detected between variables. Consider dimensionality reduction or feature selection.")
        
        # Transformation suggestions
        for col, trans in results["transformation_recommendations"].items():
            high_priority = [r for r in trans["recommendations"] if r["priority"] == "high"]
            if high_priority:
                suggestions.append(f"Column '{col}' requires {high_priority[0]['transformation']} due to {high_priority[0]['reason']}.")
        
        return suggestions
    
    def _analyze_cardinality(self, unique_count: int, total_count: int, rules: Dict) -> Dict[str, Any]:
        """Analyze cardinality of categorical variable"""
        cardinality_rules = rules["cardinality_classification"]
        
        if unique_count == cardinality_rules["binary"]:
            cardinality_type = "binary"
        elif unique_count <= cardinality_rules["low_cardinality"]:
            cardinality_type = "low_cardinality"
        elif unique_count <= cardinality_rules["medium_cardinality"]:
            cardinality_type = "medium