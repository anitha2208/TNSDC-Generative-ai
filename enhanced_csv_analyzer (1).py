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
            cardinality_type = "medium_cardinality"
        elif unique_count <= cardinality_rules["high_cardinality"]:
            cardinality_type = "high_cardinality"
        else:
            cardinality_type = "very_high_cardinality"
        
        cardinality_ratio = unique_count / total_count
        
        return {
            "cardinality_type": cardinality_type,
            "unique_count": unique_count,
            "total_count": total_count,
            "cardinality_ratio": float(cardinality_ratio),
            "interpretation": self._interpret_cardinality(cardinality_type, cardinality_ratio)
        }
    
    def _interpret_cardinality(self, cardinality_type: str, ratio: float) -> Dict[str, Any]:
        """Interpret cardinality results"""
        interpretations = {
            "binary": {
                "description": "Binary variable with two distinct values",
                "analysis_suitability": "Excellent for classification and statistical tests",
                "encoding_complexity": "Minimal"
            },
            "low_cardinality": {
                "description": "Low cardinality suitable for direct analysis",
                "analysis_suitability": "Very good for grouping and segmentation",
                "encoding_complexity": "Low"
            },
            "medium_cardinality": {
                "description": "Medium cardinality requiring careful handling",
                "analysis_suitability": "Good but may need dimensionality considerations",
                "encoding_complexity": "Moderate"
            },
            "high_cardinality": {
                "description": "High cardinality requiring advanced techniques",
                "analysis_suitability": "Challenging, needs feature engineering",
                "encoding_complexity": "High"
            },
            "very_high_cardinality": {
                "description": "Very high cardinality, likely identifier variable",
                "analysis_suitability": "Not suitable for direct analysis",
                "encoding_complexity": "Very High"
            }
        }
        
        base_interpretation = interpretations.get(cardinality_type, interpretations["medium_cardinality"])
        
        # Add ratio-based insights
        if ratio > 0.9:
            base_interpretation["uniqueness_concern"] = "Nearly unique values - consider if this is an identifier"
        elif ratio < 0.01:
            base_interpretation["diversity_concern"] = "Very low diversity - dominated by few categories"
        
        return base_interpretation
    
    def _calculate_categorical_consistency(self, data: pd.Series) -> float:
        """Calculate consistency score for categorical data"""
        value_counts = data.value_counts()
        # Measure how evenly distributed the categories are
        expected_freq = len(data) / len(value_counts)
        chi_square = ((value_counts - expected_freq) ** 2 / expected_freq).sum()
        max_chi_square = len(data) * (len(value_counts) - 1)
        
        consistency = 1 - (chi_square / max_chi_square) if max_chi_square > 0 else 1.0
        return max(0.0, min(1.0, consistency))
    
    def _calculate_categorical_quality_score(self, data: pd.Series, completeness: float, rules: Dict) -> float:
        """Calculate quality score for categorical data"""
        score = 0.0
        
        # Completeness score (40%)
        if completeness >= rules["data_quality_rules"]["excellent_completeness"]:
            score += 0.40
        elif completeness >= rules["data_quality_rules"]["good_completeness"]:
            score += 0.35
        elif completeness >= rules["data_quality_rules"]["acceptable_completeness"]:
            score += 0.30
        else:
            score += 0.40 * completeness
        
        # Consistency score (30%)
        consistency = self._calculate_categorical_consistency(data)
        score += 0.30 * consistency
        
        # Category frequency distribution (30%)
        value_counts = data.value_counts()
        min_freq = rules["data_quality_rules"]["min_category_frequency"]
        rare_categories = (value_counts / len(data) < min_freq).sum()
        rare_ratio = rare_categories / len(value_counts)
        
        frequency_score = max(0, 1 - rare_ratio)
        score += 0.30 * frequency_score
        
        return min(score, 1.0)
    
    def _detect_rare_categories(self, value_counts: pd.Series, rules: Dict) -> Dict[str, Any]:
        """Detect rare categories in categorical data"""
        min_freq = rules["data_quality_rules"]["min_category_frequency"]
        total_values = value_counts.sum()
        
        rare_categories = value_counts[value_counts / total_values < min_freq]
        
        return {
            "has_rare_categories": len(rare_categories) > 0,
            "rare_category_count": len(rare_categories),
            "rare_categories": {str(k): int(v) for k, v in rare_categories.items()},
            "rare_category_percentage": float(len(rare_categories) / len(value_counts) * 100)
        }
    
    def _assess_category_balance(self, value_counts: pd.Series) -> Dict[str, Any]:
        """Assess the balance of category distribution"""
        proportions = value_counts / value_counts.sum()
        
        # Calculate Gini coefficient for inequality
        sorted_props = np.sort(proportions.values)
        n = len(sorted_props)
        cumsum = np.cumsum(sorted_props)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        # Determine balance level
        if gini < 0.2:
            balance_level = "highly_balanced"
        elif gini < 0.4:
            balance_level = "moderately_balanced"
        elif gini < 0.6:
            balance_level = "slightly_imbalanced"
        else:
            balance_level = "highly_imbalanced"
        
        return {
            "gini_coefficient": float(gini),
            "balance_level": balance_level,
            "dominant_category_proportion": float(proportions.max()),
            "minority_category_proportion": float(proportions.min())
        }
    
    def _classify_categorical_business_type(self, column_name: str, rules: Dict) -> Dict[str, Any]:
        """Classify categorical column based on business context"""
        col_lower = column_name.lower()
        classification = "general_categorical"
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
            "suggested_use_cases": self._get_categorical_use_cases(classification)
        }
    
    def _get_categorical_use_cases(self, business_type: str) -> List[str]:
        """Get suggested use cases based on business type"""
        use_cases_map = {
            "identifier": ["Record Linking", "Data Deduplication", "Primary Key Analysis"],
            "demographic": ["Customer Segmentation", "Market Analysis", "Persona Development"],
            "geographic": ["Location Intelligence", "Regional Analysis", "Geographic Segmentation"],
            "temporal": ["Time Series Analysis", "Seasonal Patterns", "Period Comparisons"],
            "product": ["Product Analysis", "Category Performance", "Brand Studies"],
            "behavioral": ["Behavioral Segmentation", "Pattern Analysis", "User Journey Mapping"],
            "general_categorical": ["Classification Analysis", "Grouping Studies", "Category Comparisons"]
        }
        return use_cases_map.get(business_type, use_cases_map["general_categorical"])
    
    def _get_encoding_recommendations(self, unique_count: int, value_counts: pd.Series, rules: Dict) -> Dict[str, Any]:
        """Get encoding recommendations for categorical data"""
        encoding_rules = rules["encoding_strategies"]
        recommendations = []
        
        # Primary encoding recommendation
        if unique_count == 2:
            primary_encoding = {
                "method": "binary_encoding",
                "reason": "Binary variable - simple 0/1 encoding",
                "complexity": "minimal",
                "memory_impact": "very_low"
            }
        elif unique_count <= encoding_rules["one_hot_max_categories"]:
            primary_encoding = {
                "method": "one_hot_encoding",
                "reason": f"Low cardinality ({unique_count} categories) suitable for one-hot encoding",
                "complexity": "low",
                "memory_impact": "low"
            }
        elif unique_count <= encoding_rules["label_encoding_max_categories"]:
            primary_encoding = {
                "method": "label_encoding",
                "reason": f"Medium cardinality ({unique_count} categories) suitable for label encoding",
                "complexity": "medium",
                "memory_impact": "very_low"
            }
        else:
            primary_encoding = {
                "method": "target_encoding",
                "reason": f"High cardinality ({unique_count} categories) requires target encoding",
                "complexity": "high",
                "memory_impact": "low"
            }
        
        # Alternative encodings
        alternatives = []
        
        if unique_count > encoding_rules["frequency_encoding_threshold"]:
            alternatives.append({
                "method": "frequency_encoding",
                "reason": "Very high cardinality - frequency encoding can reduce dimensionality"
            })
        
        if len(value_counts) > 10:
            rare_threshold = 0.01
            rare_categories = (value_counts / value_counts.sum() < rare_threshold).sum()
            if rare_categories > 0:
                alternatives.append({
                    "method": "rare_category_grouping",
                    "reason": f"{rare_categories} rare categories could be grouped together"
                })
        
        return {
            "primary_recommendation": primary_encoding,
            "alternative_methods": alternatives,
            "dimensionality_impact": {
                "one_hot_dimensions": unique_count if unique_count <= 50 else "too_many",
                "label_encoding_dimensions": 1,
                "recommended_dimensions": 1 if primary_encoding["method"] != "one_hot_encoding" else unique_count
            }
        }
    
    def _analyze_category_distribution(self, value_counts: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution of categories"""
        total = value_counts.sum()
        proportions = value_counts / total
        
        # Calculate distribution metrics
        entropy = self._calculate_entropy(value_counts)
        
        # Identify distribution patterns
        top_category_prop = proportions.iloc[0]
        bottom_category_prop = proportions.iloc[-1]
        
        if top_category_prop > 0.8:
            pattern = "dominant_single_category"
        elif top_category_prop > 0.5:
            pattern = "majority_single_category"
        elif len(proportions) <= 5 and all(p > 0.1 for p in proportions):
            pattern = "fairly_uniform"
        else:
            pattern = "mixed_distribution"
        
        return {
            "distribution_pattern": pattern,
            "entropy": float(entropy),
            "concentration_ratio": float(proportions.head(3).sum()),  # Top 3 categories
            "diversity_index": float(1 - sum(p**2 for p in proportions)),  # Simpson's diversity
            "category_tiers": {
                "dominant": list(value_counts[proportions > 0.3].index[:3]),
                "common": list(value_counts[(proportions >= 0.05) & (proportions <= 0.3)].index),
                "rare": list(value_counts[proportions < 0.05].index)
            }
        }
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy for categorical distribution"""
        proportions = value_counts / value_counts.sum()
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        return entropy
    
    def _generate_categorical_preprocessing_suggestions(self, results: Dict, rules: Dict) -> List[str]:
        """Generate preprocessing suggestions for categorical data"""
        suggestions = []
        
        # Quality-based suggestions
        for col, quality in results["data_quality_assessment"].items():
            if quality["quality_score"] < 0.7:
                suggestions.append(f"Column '{col}' has quality issues (score: {quality['quality_score']:.2f}). Consider data cleaning.")
            
            if quality["has_rare_categories"]["has_rare_categories"]:
                rare_count = quality["has_rare_categories"]["rare_category_count"]
                suggestions.append(f"Column '{col}' has {rare_count} rare categories. Consider grouping or removal.")
        
        # Cardinality-based suggestions
        for col, cardinality in results["cardinality_analysis"].items():
            if cardinality["cardinality_type"] == "very_high_cardinality":
                suggestions.append(f"Column '{col}' has very high cardinality ({cardinality['unique_count']} categories). Consider feature engineering or dimensionality reduction.")
        
        # Encoding suggestions
        for col, encoding in results["encoding_recommendations"].items():
            if encoding["primary_recommendation"]["complexity"] == "high":
                suggestions.append(f"Column '{col}' requires advanced encoding ({encoding['primary_recommendation']['method']}) due to high cardinality.")
        
        return suggestions
    
    def perform_comprehensive_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive analysis with all predefined rules"""
        if self.df is None:
            raise HTTPException(status_code=400, detail="No data loaded for analysis")
        
        # Start with data separation
        separation_results = self.separate_numerical_categorical_data()
        
        # Apply predefined rules to numerical data
        numerical_analysis = self.apply_predefined_rules_to_numerical_data(
            separation_results["numerical_columns"]
        )
        
        # Apply predefined rules to categorical data
        categorical_analysis = self.apply_predefined_rules_to_categorical_data(
            separation_results["categorical_columns"]
        )
        
        # Overall dataset assessment
        overall_assessment = self._generate_overall_dataset_assessment(
            separation_results, numerical_analysis, categorical_analysis
        )
        
        # Compile final results
        comprehensive_results = {
            "analysis_metadata": {
                "analysis_id": self.analysis_id,
                "timestamp": datetime.now().isoformat(),
                "file_name": self.file_name,
                "input_file_path": str(self.input_file_path)
            },
            "data_separation": separation_results,
            "numerical_data_analysis": numerical_analysis,
            "categorical_data_analysis": categorical_analysis,
            "overall_assessment": overall_assessment,
            "predefined_rules_applied": {
                "numerical_rules": list(self.predefined_rules["numerical_data_rules"].keys()),
                "categorical_rules": list(self.predefined_rules["categorical_data_rules"].keys()),
                "validation_rules": list(self.predefined_rules["data_validation_rules"].keys())
            }
        }
        
        self.analysis_results = comprehensive_results
        return comprehensive_results
    
    def _generate_overall_dataset_assessment(self, separation: Dict, numerical: Dict, categorical: Dict) -> Dict[str, Any]:
        """Generate overall dataset assessment"""
        validation_rules = self.predefined_rules["data_validation_rules"]
        
        # Basic dataset metrics
        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        missing_percentage = (self.df.isnull().sum().sum() / (total_rows * total_cols)) * 100
        duplicate_percentage = (self.df.duplicated().sum() / total_rows) * 100
        
        # Quality assessment
        dataset_quality = {
            "overall_completeness": float(100 - missing_percentage),
            "duplicate_percentage": float(duplicate_percentage),
            "size_assessment": self._assess_dataset_size(total_rows, validation_rules),
            "complexity_assessment": self._assess_dataset_complexity(separation),
            "quality_grade": self._calculate_overall_quality_grade(100 - missing_percentage, duplicate_percentage)
        }
        
        # Readiness assessment
        analysis_readiness = {
            "ready_for_analysis": self._assess_analysis_readiness(separation, numerical, categorical, validation_rules),
            "recommended_next_steps": self._generate_next_steps(separation, numerical, categorical),
            "potential_challenges": self._identify_potential_challenges(numerical, categorical),
            "analysis_opportunities": self._identify_analysis_opportunities(separation, numerical, categorical)
        }
        
        return {
            "dataset_summary": {
                "total_records": total_rows,
                "total_features": total_cols,
                "numerical_features": separation["total_numerical"],
                "categorical_features": separation["total_categorical"],
                "datetime_features": separation["total_datetime"]
            },
            "quality_assessment": dataset_quality,
            "analysis_readiness": analysis_readiness,
            "preprocessing_priority": self._determine_preprocessing_priority(numerical, categorical)
        }
    
    def _assess_dataset_size(self, row_count: int, rules: Dict) -> Dict[str, Any]:
        """Assess dataset size adequacy"""
        size_rules = rules["dataset_quality"]
        
        if row_count >= size_rules["recommended_rows"]:
            size_category = "large"
            adequacy = "excellent"
        elif row_count >= size_rules["minimum_rows"]:
            size_category = "medium"
            adequacy = "adequate"
        else:
            size_category = "small"
            adequacy = "limited"
        
        return {
            "size_category": size_category,
            "adequacy_for_analysis": adequacy,
            "row_count": row_count,
            "statistical_power": "high" if row_count >= 1000 else "medium" if row_count >= 100 else "low"
        }
    
    def _assess_dataset_complexity(self, separation: Dict) -> Dict[str, Any]:
        """Assess dataset complexity"""
        total_features = separation["total_numerical"] + separation["total_categorical"]
        
        if total_features > 50:
            complexity = "high"
        elif total_features > 20:
            complexity = "medium"
        else:
            complexity = "low"
        
        return {
            "complexity_level": complexity,
            "feature_diversity": "mixed" if separation["total_numerical"] > 0 and separation["total_categorical"] > 0 else "single_type",
            "dimensionality_concerns": total_features > 100
        }
    
    def _calculate_overall_quality_grade(self, completeness: float, duplicate_pct: float) -> str:
        """Calculate overall quality grade"""
        score = 0
        
        # Completeness score
        if completeness >= 95:
            score += 40
        elif completeness >= 85:
            score += 35
        elif completeness >= 70:
            score += 25
        else:
            score += int(completeness * 0.4)
        
        # Duplicate score
        if duplicate_pct <= 1:
            score += 30
        elif duplicate_pct <= 5:
            score += 25
        elif duplicate_pct <= 10:
            score += 15
        else:
            score += 5
        
        # Base score for having data
        score += 30
        
        if score >= 90:
            return "A - Excellent"
        elif score >= 80:
            return "B - Very Good"
        elif score >= 70:
            return "C - Good"
        elif score >= 60:
            return "D - Fair"
        else:
            return "F - Poor"
    
    def _assess_analysis_readiness(self, separation: Dict, numerical: Dict, categorical: Dict, rules: Dict) -> Dict[str, Any]:
        """Assess if dataset is ready for analysis"""
        readiness_score = 0
        max_score = 100
        issues = []
        
        # Data volume check (20 points)
        if len(self.df) >= rules["dataset_quality"]["minimum_rows"]:
            readiness_score += 20
        else:
            issues.append(f"Insufficient data volume ({len(self.df)} rows, minimum {rules['dataset_quality']['minimum_rows']})")
        
        # Data variety check (20 points)
        if separation["total_numerical"] > 0 and separation["total_categorical"] > 0:
            readiness_score += 20
        elif separation["total_numerical"] > 0 or separation["total_categorical"] > 0:
            readiness_score += 15
        else:
            issues.append("No numerical or categorical data found")
        
        # Data quality check (30 points)
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_pct <= 5:
            readiness_score += 30
        elif missing_pct <= 15:
            readiness_score += 20
        elif missing_pct <= 30:
            readiness_score += 10
        else:
            issues.append(f"High missing data percentage ({missing_pct:.1f}%)")
        
        # Duplicate check (15 points)
        duplicate_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        if duplicate_pct <= rules["dataset_quality"]["max_duplicate_percentage"] * 100:
            readiness_score += 15
        else:
            issues.append(f"High duplicate percentage ({duplicate_pct:.1f}%)")
        
        # Feature engineering check (15 points)
        if numerical and "preprocessing_suggestions" in numerical:
            high_priority_issues = sum(1 for suggestion in numerical.get("preprocessing_suggestions", []) if "high" in suggestion.lower())
            if high_priority_issues == 0:
                readiness_score += 15
            elif high_priority_issues <= 2:
                readiness_score += 10
            else:
                issues.append("Multiple high-priority preprocessing issues in numerical data")
        
        readiness_level = "ready" if readiness_score >= 80 else "needs_preprocessing" if readiness_score >= 60 else "needs_significant_work"
        
        return {
            "readiness_score": readiness_score,
            "readiness_level": readiness_level,
            "blocking_issues": issues,
            "can_proceed_with_analysis": readiness_score >= 60
        }
    
    def _generate_next_steps(self, separation: Dict, numerical: Dict, categorical: Dict) -> List[str]:
        """Generate recommended next steps"""
        next_steps = []
        
        # Data quality steps
        if numerical and "preprocessing_suggestions" in numerical:
            high_priority = [s for s in numerical["preprocessing_suggestions"] if "high" in s.lower() or "quality" in s.lower()]
            for suggestion in high_priority[:3]:  # Top 3 suggestions
                next_steps.append(f"Address numerical data issue: {suggestion}")
        
        if categorical and "preprocessing_suggestions" in categorical:
            high_priority = [s for s in categorical["preprocessing_suggestions"] if "high" in s.lower() or "quality" in s.lower()]
            for suggestion in high_priority[:3]:  # Top 3 suggestions
                next_steps.append(f"Address categorical data issue: {suggestion}")
        
        # Analysis steps
        if separation["total_numerical"] > 1:
            next_steps.append("Perform correlation analysis on numerical variables")
        
        if separation["total_categorical"] > 0:
            next_steps.append("Conduct categorical data encoding and feature engineering")
        
        if separation["total_numerical"] > 0 and separation["total_categorical"] > 0:
            next_steps.append("Explore relationships between numerical and categorical variables")
        
        return next_steps[:5]  # Limit to top 5 steps
    
    def _identify_potential_challenges(self, numerical: Dict, categorical: Dict) -> List[str]:
        """Identify potential challenges for analysis"""
        challenges = []
        
        # Numerical data challenges
        if numerical and numerical != {"message": "No numerical columns available for analysis"}:
            if "correlation_analysis" in numerical and numerical["correlation_analysis"].get("multicollinearity_risk"):
                challenges.append("Multicollinearity detected in numerical variables")
            
            outlier_columns = []
            for col, quality in numerical.get("data_quality_assessment", {}).items():
                if quality.get("has_outliers", {}).get("iqr_outlier_percentage", 0) > 15:
                    outlier_columns.append(col)
            
            if outlier_columns:
                challenges.append(f"Significant outliers in columns: {', '.join(outlier_columns[:3])}")
        
        # Categorical data challenges
        if categorical and categorical != {"message": "No categorical columns available for analysis"}:
            high_cardinality_cols = []
            for col, cardinality in categorical.get("cardinality_analysis", {}).items():
                if cardinality.get("cardinality_type") == "very_high_cardinality":
                    high_cardinality_cols.append(col)
            
            if high_cardinality_cols:
                challenges.append(f"Very high cardinality in columns: {', '.join(high_cardinality_cols[:3])}")
        
        return challenges
    
    def _identify_analysis_opportunities(self, separation: Dict, numerical: Dict, categorical: Dict) -> List[str]:
        """Identify analysis opportunities"""
        opportunities = []
        
        # Based on data composition
        if separation["total_numerical"] >= 3:
            opportunities.append("Multi-variate statistical analysis and predictive modeling")
        
        if separation["total_categorical"] >= 2:
            opportunities.append("Market segmentation and customer profiling analysis")
        
        if separation["total_numerical"] > 0 and separation["total_categorical"] > 0:
            opportunities.append("Mixed-type analysis and feature interaction studies")
        
        # Based on data quality
        total_quality_score = 0
        quality_count = 0
        
        if numerical and "data_quality_assessment" in numerical:
            for col_quality in numerical["data_quality_assessment"].values():
                total_quality_score += col_quality.get("quality_score", 0)
                quality_count += 1
        
        if categorical and "data_quality_assessment" in categorical:
            for col_quality in categorical["data_quality_assessment"].values():
                total_quality_score += col_quality.get("quality_score", 0)
                quality_count += 1
        
        if quality_count > 0:
            avg_quality = total_quality_score / quality_count
            if avg_quality > 0.8:
                opportunities.append("High-quality data suitable for advanced analytics and machine learning")
        
        return opportunities
    
    def _determine_preprocessing_priority(self, numerical: Dict, categorical: Dict) -> List[Dict[str, Any]]:
        """Determine preprocessing priority order"""
        priority_tasks = []
        
        # High priority tasks
        if numerical and "preprocessing_suggestions" in numerical:
            for suggestion in numerical["preprocessing_suggestions"]:
                if "quality" in suggestion.lower() or "outlier" in suggestion.lower():
                    priority_tasks.append({
                        "task": suggestion,
                        "priority": "high",
                        "data_type": "numerical",
                        "estimated_impact": "high"
                    })
        
        if categorical and "preprocessing_suggestions" in categorical:
            for suggestion in categorical["preprocessing_suggestions"]:
                if "quality" in suggestion.lower() or "rare" in suggestion.lower():
                    priority_tasks.append({
                        "task": suggestion,
                        "priority": "high",
                        "data_type": "categorical",
                        "estimated_impact": "high"
                    })
        
        # Sort by priority and impact
        priority_order = {"high": 3, "medium": 2, "low": 1}
        priority_tasks.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
        
        return priority_tasks[:10]  # Top 10 priority tasks
    
    def generate_final_json_output(self) -> Dict[str, Any]:
        """Generate final JSON output with all analysis results"""
        if not self.analysis_results:
            return {"error": "No analysis results available"}
        
        final_output = {
            "analysis_summary": {
                "analysis_id": self.analysis_id,
                "timestamp": datetime.now().isoformat(),
                "original_filename": self.file_name,
                "input_csv_path": str(self.input_file_path),
                "analyzer_version": "2.0.0"
            },
            "dataset_overview": {
                "total_records": len(self.df),
                "total_features": len(self.df.columns),
                "data_types": self.analysis_results["data_separation"]["separation_summary"]["data_type_distribution"]
            },
            "data_separation_results": self.analysis_results["data_separation"],
            "numerical_data_analysis": self.analysis_results["numerical_data_analysis"],
            "categorical_data_analysis": self.analysis_results["categorical_data_analysis"],
            "overall_dataset_assessment": self.analysis_results["overall_assessment"],
            "predefined_rules_summary": {
                "rules_applied": self.analysis_results["predefined_rules_applied"],
                "total_rules_count": len(self.analysis_results["predefined_rules_applied"]["numerical_rules"]) + 
                                  len(self.analysis_results["predefined_rules_applied"]["categorical_rules"]) +
                                  len(self.analysis_results["predefined_rules_applied"]["validation_rules"])
            },
            "analysis_completion_status": "success"
        }
        
        return final_output

# FastAPI endpoints
@app.post("/analyze-csv")
async def analyze_uploaded_csv(file: UploadFile = File(...)):
    """
    Upload CSV file, generate unique ID, save as input.