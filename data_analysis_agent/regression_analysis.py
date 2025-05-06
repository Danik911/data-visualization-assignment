import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from config import get_config

class RegressionModel:
    """
    A class to implement and evaluate linear regression models for any dataset.
    This class handles both full dataset models and category-specific models.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None, predictor_column: Optional[str] = None):
        """
        Initialize the RegressionModel class with automatic target and predictor detection.
        
        Args:
            df: The DataFrame containing the data
            target_column: The target variable (dependent variable), or None to auto-detect
            predictor_column: The predictor variable (independent variable), or None to auto-detect
        """
        self.df = df.copy()
        self.config = get_config()
        
        # Auto-detect target and predictor columns if not provided
        if target_column is None:
            self.target_column = self._detect_target_column()
            print(f"[REGRESSION] Auto-detected target column: {self.target_column}")
        else:
            self.target_column = target_column
            
        if predictor_column is None:
            self.predictor_column = self._detect_predictor_column()
            print(f"[REGRESSION] Auto-detected predictor column: {self.predictor_column}")
        else:
            self.predictor_column = predictor_column
            
        self.models = {}
        self.model_results = {}
        self.sklearn_models = {}
        
        # Check if the necessary columns exist
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        if self.predictor_column not in self.df.columns:
            raise ValueError(f"Predictor column '{self.predictor_column}' not found in DataFrame")
    
    def _detect_target_column(self) -> str:
        """
        Detect the most appropriate target column based on dataset properties.
        
        Returns:
            The name of the detected target column
        """
        # First check if config has a recommended target from dataset analysis
        target = self.config.get_target_column()
        if target and target in self.df.columns:
            return target
            
        # If no target in config, try to detect one based on column names
        
        # Common target column patterns for various domains
        target_patterns = [
            # Housing/real estate domain
            ['price', 'saleprice', 'value', 'cost', 'worth', 'appraisal'],
            # Transportation domain
            ['time', 'duration', 'delay', 'commute'],
            # Health/medical domain
            ['mortality', 'survival', 'recovery', 'readmission'],
            # Academic domain
            ['score', 'grade', 'performance', 'gpa'],
            # Business domain
            ['revenue', 'sales', 'profit', 'income', 'earnings']
        ]
        
        # Check for columns matching target patterns
        for pattern_group in target_patterns:
            for pattern in pattern_group:
                for col in self.df.columns:
                    if pattern in col.lower():
                        print(f"[REGRESSION] Detected likely target column '{col}' based on name pattern '{pattern}'")
                        return col
        
        # If no match in column names, use statistical properties
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Exclude ID-like columns (high cardinality relative to dataframe size)
            potential_targets = []
            for col in numeric_cols:
                # Skip likely ID columns
                if (col.lower().endswith('id') or 
                   'id_' in col.lower() or 
                   col.lower() == 'id' or 
                   self.df[col].nunique() / len(self.df) > 0.9):
                    continue
                potential_targets.append(col)
            
            if potential_targets:
                # Calculate statistics for each potential target
                target_scores = {}
                
                for col in potential_targets:
                    score = 0
                    
                    # Columns with higher variance relative to their mean are better targets
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    if mean != 0:
                        cv = std / abs(mean)  # Coefficient of variation
                        score += min(cv * 10, 50)  # Cap the score
                    
                    # Columns with non-zero skewness might be better targets
                    try:
                        skew = self.df[col].skew()
                        score += min(abs(skew) * 5, 20)  # Some skew is typical of target variables
                    except:
                        pass
                    
                    # Columns at the end of the DataFrame are often targets
                    col_index = list(self.df.columns).index(col)
                    if col_index > len(self.df.columns) * 0.7:
                        score += 15
                    
                    target_scores[col] = score
                
                # Select column with highest score
                if target_scores:
                    best_target = max(target_scores.items(), key=lambda x: x[1])[0]
                    print(f"[REGRESSION] Selected target column '{best_target}' based on statistical properties")
                    return best_target
                
                # If scoring doesn't yield a clear winner, use the last numeric column
                # (often targets are at the end of the dataset)
                return numeric_cols[-1]
                
        # If we still don't have a target, use the first numeric column
        if numeric_cols:
            return numeric_cols[0]
                
        # If no suitable target found, raise an error
        raise ValueError("Could not detect an appropriate target column. Please specify one explicitly.")
    
    def _detect_predictor_column(self) -> str:
        """
        Detect the most appropriate predictor column based on dataset properties and target.
        
        Returns:
            The name of the detected predictor column
        """
        # First check if config has recommended predictors
        predictors = self.config.get_predictor_columns()
        if predictors and any(pred in self.df.columns for pred in predictors):
            for pred in predictors:
                if pred in self.df.columns and pred != self.target_column:
                    return pred
        
        # If no predictors in config, try to detect one based on domain knowledge
        domain = getattr(self.config, 'dataset_properties', {}).get('detected_domain')
        
        # Dictionary of domain-specific predictor patterns
        domain_patterns = {
            'housing': ['area', 'sqft', 'squarefeet', 'bedrooms', 'bathrooms', 'garage', 'stories', 'year', 'age', 'lot'],
            'transportation': ['distance', 'miles', 'kilometers', 'km', 'speed', 'velocity'],
            'academic': ['hours', 'attendance', 'study', 'classes', 'credits'],
            'health': ['weight', 'height', 'bmi', 'age', 'dosage', 'exercise', 'steps'],
            'business': ['marketing', 'advertising', 'employees', 'assets', 'investment']
        }
        
        # Check domain-specific predictors
        if domain in domain_patterns:
            for term in domain_patterns[domain]:
                for col in self.df.columns:
                    if term in col.lower() and col != self.target_column:
                        print(f"[REGRESSION] Detected likely predictor column '{col}' based on domain pattern '{term}'")
                        return col
        
        # If no domain-specific match, check common patterns across domains
        general_predictors = ['year', 'amount', 'count', 'number', 'size', 'age', 'rate']
        for term in general_predictors:
            for col in self.df.columns:
                if term in col.lower() and col != self.target_column:
                    print(f"[REGRESSION] Detected likely predictor column '{col}' based on general pattern '{term}'")
                    return col
        
        # If no name-based match, use statistical approach
        numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                       if col != self.target_column and not (
                           col.lower().endswith('id') or 
                           'id_' in col.lower() or 
                           col.lower() == 'id' or 
                           self.df[col].nunique() / len(self.df) > 0.9  # Exclude likely IDs
                       )]
        
        if numeric_cols:
            # Calculate correlations with target
            correlations = {}
            for col in numeric_cols:
                # Skip columns with too many missing values
                if self.df[col].isna().sum() / len(self.df) > 0.5:
                    continue
                
                try:
                    corr = self.df[[self.target_column, col]].corr().iloc[0, 1]
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)  # Use absolute correlation
                except Exception as e:
                    print(f"[REGRESSION] Error calculating correlation for '{col}': {str(e)}")
                    continue
            
            if correlations:
                # Return column with highest absolute correlation to target
                best_predictor = max(correlations.items(), key=lambda x: x[1])[0]
                print(f"[REGRESSION] Selected predictor '{best_predictor}' with correlation of {correlations[best_predictor]:.4f}")
                return best_predictor
            else:
                # If no correlations, return first non-target numeric column
                for col in numeric_cols:
                    if col != self.target_column:
                        return col
                
        # If we still have no predictor but have numeric columns, use the first one that's not the target
        all_numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in all_numeric:
            if col != self.target_column:
                return col
                
        # If no suitable predictor found, raise an error
        raise ValueError("Could not detect an appropriate predictor column. Please specify one explicitly.")
    
    def fit_full_dataset_model(self) -> Dict[str, Any]:
        """
        Fit a linear regression model on the full dataset.
        
        Returns:
            Dictionary with regression metrics and model information
        """
        print(f"[REGRESSION] Fitting full dataset model: {self.target_column} ~ {self.predictor_column}")
        
        # Drop rows with missing values in target or predictor
        clean_df = self.df.dropna(subset=[self.target_column, self.predictor_column])
        
        # Extract X and y
        X = clean_df[[self.predictor_column]]
        y = clean_df[self.target_column]
        
        # Add constant for statsmodels
        X_sm = sm.add_constant(X)
        
        # Fit statsmodels OLS for detailed statistics
        model = sm.OLS(y, X_sm).fit()
        
        # Store the model
        self.models['full_dataset'] = model
        
        # Fit sklearn model for prediction
        skl_model = LinearRegression()
        skl_model.fit(X, y)
        self.sklearn_models['full_dataset'] = skl_model
        
        # Calculate metrics
        r_squared = model.rsquared
        adjusted_r_squared = model.rsquared_adj
        rmse = np.sqrt(model.mse_resid)
        mae = np.mean(np.abs(model.resid))
        
        # Calculate correlation coefficient
        correlation = clean_df[[self.target_column, self.predictor_column]].corr().iloc[0, 1]
        
        # Extract model coefficients and p-values
        coef = model.params[1]
        intercept = model.params[0]
        p_value = model.pvalues[1]
        
        # Get confidence intervals for coefficients
        conf_int = model.conf_int(alpha=0.05)
        coef_ci_low, coef_ci_high = conf_int.iloc[1, 0], conf_int.iloc[1, 1]
        intercept_ci_low, intercept_ci_high = conf_int.iloc[0, 0], conf_int.iloc[0, 1]
        
        # Check statistical significance
        is_significant = p_value < 0.05
        
        results = {
            'model_type': 'full_dataset',
            'metrics': {
                'r': correlation,
                'r_squared': r_squared,
                'adjusted_r_squared': adjusted_r_squared,
                'rmse': rmse,
                'mae': mae
            },
            'coefficients': {
                'intercept': intercept,
                'coefficient': coef,
                'p_value': p_value,
                'is_significant': is_significant
            },
            'confidence_intervals': {
                'intercept_ci': [intercept_ci_low, intercept_ci_high],
                'coefficient_ci': [coef_ci_low, coef_ci_high]
            },
            'formula': f"{self.target_column} = {intercept:.2f} + {coef:.2f} * {self.predictor_column}"
        }
        
        # Store results
        self.model_results['full_dataset'] = results
        
        print(f"[REGRESSION] Full dataset model fitted with R² = {r_squared:.4f}")
        return results
    
    def fit_category_specific_models(self, category_column: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Fit separate linear regression models for each category.
        
        Args:
            category_column: The column containing category information, or None to auto-detect
            
        Returns:
            Dictionary with model results for each category
        """
        # Auto-detect category column if not specified
        if category_column is None:
            category_column = self._detect_category_column()
            if category_column is None:
                print(f"[REGRESSION] No suitable category column detected. Skipping category-specific models.")
                return {}
        
        if category_column not in self.df.columns:
            raise ValueError(f"Category column '{category_column}' not found in DataFrame")
        
        print(f"[REGRESSION] Fitting category-specific models for each value in '{category_column}'")
        
        all_category_results = {}
        
        # For each unique category value
        for category in self.df[category_column].unique():
            # Filter data for this category
            category_df = self.df[self.df[category_column] == category]
            
            # Only proceed if we have enough data points (at least 3 for regression)
            if len(category_df) < 3:
                print(f"[REGRESSION] Not enough data points for category '{category}' (count: {len(category_df)})")
                continue
            
            # Drop rows with missing values in target or predictor
            clean_category_df = category_df.dropna(subset=[self.target_column, self.predictor_column])
            
            if len(clean_category_df) < 3:
                print(f"[REGRESSION] Not enough clean data points for category '{category}' after dropping NA values")
                continue
            
            print(f"[REGRESSION] Fitting model for category '{category}' with {len(clean_category_df)} data points")
            
            # Extract X and y
            X = clean_category_df[[self.predictor_column]]
            y = clean_category_df[self.target_column]
            
            # Add constant for statsmodels
            X_sm = sm.add_constant(X)
            
            try:
                # Fit statsmodels OLS
                model = sm.OLS(y, X_sm).fit()
                
                # Store the model
                model_key = f"category_{category}"
                self.models[model_key] = model
                
                # Fit sklearn model for prediction
                skl_model = LinearRegression()
                skl_model.fit(X, y)
                self.sklearn_models[model_key] = skl_model
                
                # Calculate metrics
                r_squared = model.rsquared
                adjusted_r_squared = model.rsquared_adj
                rmse = np.sqrt(model.mse_resid)
                mae = np.mean(np.abs(model.resid))
                
                # Calculate correlation coefficient
                correlation = clean_category_df[[self.target_column, self.predictor_column]].corr().iloc[0, 1]
                
                # Extract model coefficients and p-values
                coef = model.params[1]
                intercept = model.params[0]
                p_value = model.pvalues[1]
                
                # Get confidence intervals for coefficients
                conf_int = model.conf_int(alpha=0.05)
                coef_ci_low, coef_ci_high = conf_int.iloc[1, 0], conf_int.iloc[1, 1]
                intercept_ci_low, intercept_ci_high = conf_int.iloc[0, 0], conf_int.iloc[0, 1]
                
                # Check statistical significance
                is_significant = p_value < 0.05
                
                results = {
                    'model_type': f"category_{category}",
                    'category': category,
                    'sample_size': len(clean_category_df),
                    'metrics': {
                        'r': correlation,
                        'r_squared': r_squared,
                        'adjusted_r_squared': adjusted_r_squared,
                        'rmse': rmse,
                        'mae': mae
                    },
                    'coefficients': {
                        'intercept': intercept,
                        'coefficient': coef,
                        'p_value': p_value,
                        'is_significant': is_significant
                    },
                    'confidence_intervals': {
                        'intercept_ci': [intercept_ci_low, intercept_ci_high],
                        'coefficient_ci': [coef_ci_low, coef_ci_high]
                    },
                    'formula': f"{self.target_column} = {intercept:.2f} + {coef:.2f} * {self.predictor_column}"
                }
                
                # Store results
                self.model_results[model_key] = results
                all_category_results[category] = results
                
                print(f"[REGRESSION] Category '{category}' model fitted with R² = {r_squared:.4f}")
            
            except Exception as e:
                print(f"[REGRESSION] Error fitting model for category '{category}': {str(e)}")
                all_category_results[category] = {"error": str(e), "category": category, "sample_size": len(clean_category_df)}
        
        return all_category_results
    
    def _detect_category_column(self) -> Optional[str]:
        """
        Detect a suitable categorical column for category-specific models.
        
        Returns:
            The name of a detected categorical column, or None if no suitable column is found
        """
        # Get detected domain
        domain = getattr(self.config, 'dataset_properties', {}).get('detected_domain')
        
        # Domain-specific checks
        if domain == 'transportation':
            # For transportation data, look for 'mode' column
            for col in self.df.columns:
                if col.lower() in ['mode', 'transport_mode', 'transportation', 'vehicle']:
                    return col
        
        elif domain == 'housing':
            # For housing data, look for neighborhood, property type, etc.
            for col in self.df.columns:
                if col.lower() in ['neighborhood', 'suburb', 'zip', 'zipcode', 'type', 'property_type']:
                    if self.df[col].nunique() <= 20:  # Not too many categories
                        return col
        
        # Generic checks: find categorical columns
        object_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Add numeric columns with few unique values
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].nunique() <= 10 and col != self.target_column and col != self.predictor_column:
                object_cols.append(col)
        
        # Filter to columns with a reasonable number of categories
        suitable_cols = []
        for col in object_cols:
            n_unique = self.df[col].nunique()
            # Column should have 2-20 categories and each category should have multiple rows
            if 2 <= n_unique <= 20:
                suitable_cols.append((col, n_unique))
        
        if suitable_cols:
            # Prioritize columns with fewer categories (3-7 is ideal)
            sorted_cols = sorted(suitable_cols, key=lambda x: abs(x[1] - 5))
            return sorted_cols[0][0]
            
        return None
    
    def generate_regression_plots(self, output_dir: str = "plots/regression") -> List[str]:
        """
        Generate regression plots for all fitted models.
        
        Args:
            output_dir: Directory to save plots in
            
        Returns:
            List of saved plot file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = []
        
        # 1. Plot for full dataset model
        if 'full_dataset' in self.models:
            plt.figure(figsize=(10, 6))
            
            # Scatter plot of actual data
            sns.scatterplot(data=self.df, x=self.predictor_column, y=self.target_column, alpha=0.6, label='Data points')
            
            # Add regression line
            model = self.models['full_dataset']
            intercept = model.params[0]
            coef = model.params[1]
            
            # Create range of x values for prediction line
            x_range = np.linspace(self.df[self.predictor_column].min(), self.df[self.predictor_column].max(), 100)
            y_pred = intercept + coef * x_range
            
            plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'Regression Line: y = {intercept:.2f} + {coef:.2f}x')
            
            # Add R-squared to plot
            r_squared = self.model_results['full_dataset']['metrics']['r_squared']
            plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))
            
            plt.title(f'Linear Regression: {self.target_column} vs {self.predictor_column} (Full Dataset)')
            plt.xlabel(self.predictor_column)
            plt.ylabel(self.target_column)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            full_plot_path = os.path.join(output_dir, "full_dataset_regression.png")
            plt.savefig(full_plot_path)
            plt.close()
            plot_paths.append(full_plot_path)
            print(f"[REGRESSION] Saved plot: {full_plot_path}")
        
        # 2. Plot for category-specific models
        category_keys = [key for key in self.models.keys() if key.startswith('category_')]
        
        if category_keys:
            # Identify the category column name
            first_category_key = category_keys[0]
            category_name = first_category_key.replace('category_', '')
            category_column = self._detect_category_column()
            
            if category_column:
                # Create a single plot with all category-specific regression lines
                plt.figure(figsize=(12, 8))
                
                # Create scatter plot with points colored by category
                sns.scatterplot(data=self.df, x=self.predictor_column, y=self.target_column, 
                               hue=category_column, alpha=0.6)
                
                # Add regression lines for each category
                for key, model in self.models.items():
                    if key.startswith('category_'):
                        category = key.replace('category_', '')
                        intercept = model.params[0]
                        coef = model.params[1]
                        
                        # Create range of x values for prediction line
                        category_df = self.df[self.df[category_column] == category]
                        if len(category_df) > 0:
                            x_min = category_df[self.predictor_column].min()
                            x_max = category_df[self.predictor_column].max()
                            x_range = np.linspace(x_min, x_max, 100)
                            y_pred = intercept + coef * x_range
                            
                            plt.plot(x_range, y_pred, linewidth=2, label=f'{category}: y = {intercept:.2f} + {coef:.2f}x')
                
                plt.title(f'Linear Regression by {category_column}: {self.target_column} vs {self.predictor_column}')
                plt.xlabel(self.predictor_column)
                plt.ylabel(self.target_column)
                plt.grid(True, alpha=0.3)
                plt.legend(title=category_column)
                
                categories_plot_path = os.path.join(output_dir, "category_specific_regression.png")
                plt.savefig(categories_plot_path)
                plt.close()
                plot_paths.append(categories_plot_path)
                print(f"[REGRESSION] Saved plot: {categories_plot_path}")
                
                # Individual plots for each category
                for key, model in self.models.items():
                    if key.startswith('category_'):
                        category = key.replace('category_', '')
                        category_df = self.df[self.df[category_column] == category]
                        
                        plt.figure(figsize=(10, 6))
                        
                        # Scatter plot for this category
                        sns.scatterplot(data=category_df, x=self.predictor_column, y=self.target_column, alpha=0.6, label=f'{category} data points')
                        
                        # Add regression line
                        intercept = model.params[0]
                        coef = model.params[1]
                        
                        # Create range of x values for prediction line
                        x_min = category_df[self.predictor_column].min()
                        x_max = category_df[self.predictor_column].max()
                        x_range = np.linspace(x_min, x_max, 100)
                        y_pred = intercept + coef * x_range
                        
                        plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'Regression: y = {intercept:.2f} + {coef:.2f}x')
                        
                        # Add R-squared to plot
                        r_squared = self.model_results[key]['metrics']['r_squared']
                        plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
                                 bbox=dict(facecolor='white', alpha=0.8))
                        
                        plt.title(f'Linear Regression: {self.target_column} vs {self.predictor_column} (Category: {category})')
                        plt.xlabel(self.predictor_column)
                        plt.ylabel(self.target_column)
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        
                        category_plot_path = os.path.join(output_dir, f"{category.lower()}_regression.png")
                        plt.savefig(category_plot_path)
                        plt.close()
                        plot_paths.append(category_plot_path)
                        print(f"[REGRESSION] Saved plot: {category_plot_path}")
        
        return plot_paths
    
    def predict(self, predictor_values: List[float], model_type: str = 'full_dataset') -> Dict[str, Any]:
        """
        Generate predictions using the specified model.
        
        Args:
            predictor_values: List of predictor values to predict target for
            model_type: Type of model to use ('full_dataset' or 'category_XXX')
            
        Returns:
            Dictionary with predictions and intervals
        """
        if model_type not in self.sklearn_models:
            raise ValueError(f"Model type '{model_type}' not found in fitted models")
        
        # Create DataFrame from predictor values
        X_pred = pd.DataFrame({self.predictor_column: predictor_values})
        
        # Get the sklearn model for prediction
        model = self.sklearn_models[model_type]
        statsmodel = self.models[model_type]
        
        # Get point predictions
        y_pred = model.predict(X_pred)
        
        # Calculate prediction intervals using statsmodels
        X_sm = sm.add_constant(X_pred)
        
        # Get prediction statistics from statsmodels
        try:
            pred = statsmodel.get_prediction(X_sm)
            pred_intervals = pred.conf_int(alpha=0.05)  # 95% prediction interval
            
            lower_interval = pred_intervals[:, 0]
            upper_interval = pred_intervals[:, 1]
            
            results = {
                'predictor_values': predictor_values,
                'predicted_targets': y_pred.tolist(),
                'lower_intervals': lower_interval.tolist(),
                'upper_intervals': upper_interval.tolist(),
                'model_type': model_type
            }
            
            return results
        except Exception as e:
            print(f"[REGRESSION] Error calculating prediction intervals: {str(e)}")
            # Return basic predictions without intervals
            return {
                'predictor_values': predictor_values,
                'predicted_targets': y_pred.tolist(),
                'model_type': model_type,
                'error': f"Could not calculate prediction intervals: {str(e)}"
            }
    
    def get_model_summary(self, model_type: str = 'full_dataset') -> str:
        """
        Get a detailed summary of the specified model.
        
        Args:
            model_type: Type of model to summarize ('full_dataset' or 'category_XXX')
            
        Returns:
            String with model summary
        """
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not found in fitted models")
        
        return self.models[model_type].summary().as_text()
    
    def save_model_results(self, file_path: str = "reports/regression_models.json") -> None:
        """
        Save all model results to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        print(f"[REGRESSION] Attempting to save model results to {file_path}")
        
        # Make all values JSON serializable
        def make_serializable(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, (np.float64, np.float32)):
                return float(item)
            elif isinstance(item, (np.int64, np.int32)):
                return int(item)
            elif isinstance(item, np.bool_):  # Handle both Python and NumPy boolean types
                return bool(item)  # Convert to Python native boolean
            else:
                # Log unusual types that might cause serialization issues
                if not isinstance(item, (str, int, float, list, dict, type(None))):
                    print(f"[REGRESSION WARNING] Potentially non-serializable type encountered: {type(item)}")
                return item
        
        try:
            # Convert results to serializable format
            serializable_results = {}
            for key, result in self.model_results.items():
                serializable_results[key] = {}
                for k, v in result.items():
                    if isinstance(v, dict):
                        serializable_results[key][k] = {k2: make_serializable(v2) for k2, v2 in v.items()}
                    else:
                        serializable_results[key][k] = make_serializable(v)
            
            # Add dataset and model configuration
            serializable_results['model_config'] = {
                'target_column': self.target_column,
                'predictor_column': self.predictor_column,
                'dataset_rows': len(self.df),
                'dataset_columns': len(self.df.columns),
            }
            
            # Add information about how columns were detected
            serializable_results['column_detection'] = {
                'target_auto_detected': hasattr(self, '_detected_target'),
                'predictor_auto_detected': hasattr(self, '_detected_predictor'),
                'detected_domain': getattr(self.config, 'dataset_properties', {}).get('detected_domain', 'unknown')
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"[REGRESSION] Model results successfully saved to {file_path}")
        except Exception as e:
            print(f"[REGRESSION ERROR] Failed to save model results to {file_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())


def perform_regression_analysis(df: pd.DataFrame, 
                               target_column: Optional[str] = None, 
                               predictor_column: Optional[str] = None,
                               category_column: Optional[str] = None,
                               save_report: bool = True,
                               report_path: str = "reports/regression_models.json",
                               generate_plots: bool = True,
                               plots_dir: str = "plots/regression") -> Dict[str, Any]:
    """
    Perform comprehensive regression analysis on the dataset.
    
    Args:
        df: The DataFrame containing the data
        target_column: The target variable (dependent variable), or None to auto-detect
        predictor_column: The predictor variable (independent variable), or None to auto-detect
        category_column: The category column for category-specific models, or None to auto-detect
        save_report: Whether to save the report to a file
        report_path: Path to save the report
        generate_plots: Whether to generate regression plots
        plots_dir: Directory to save plots in
        
    Returns:
        Dictionary with regression analysis results
    """
    print(f"[REGRESSION] Starting regression analysis (auto-detection: target={target_column is None}, predictor={predictor_column is None})")
    
    try:
        # Create RegressionModel instance with auto-detection capabilities
        regression_model = RegressionModel(df, target_column, predictor_column)
        
        # Report detected columns
        target = regression_model.target_column
        predictor = regression_model.predictor_column
        print(f"[REGRESSION] Using target column: {target}, predictor column: {predictor}")
        
        # Fit full dataset model
        full_model_results = regression_model.fit_full_dataset_model()
        
        # Fit category-specific models
        category_results = {}
        category_models_available = regression_model._detect_category_column() is not None or category_column is not None
        if category_models_available:
            category_results = regression_model.fit_category_specific_models(category_column)
        
        # Generate plots if requested
        plot_paths = []
        if generate_plots:
            plot_paths = regression_model.generate_regression_plots(output_dir=plots_dir)
        
        # Save model results if requested
        if save_report:
            regression_model.save_model_results(file_path=report_path)
        
        # Return comprehensive results
        analysis_results = {
            'full_model': full_model_results,
            'category_models': category_results,
            'plot_paths': plot_paths,
            'report_path': report_path if save_report else None,
            'target_column': target,
            'predictor_column': predictor,
        }
        
        # Add success message
        analysis_results['status'] = 'success'
        
        print(f"[REGRESSION] Regression analysis completed successfully")
        return analysis_results
    
    except Exception as e:
        import traceback
        print(f"[REGRESSION ERROR] Error in regression analysis: {str(e)}")
        print(traceback.format_exc())
        return {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }