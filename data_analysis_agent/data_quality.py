"""
Data Quality Assessment and Cleaning Module

This module provides functionality for comprehensive data quality assessment and cleaning,
implementing the requirements from the project plan:
- Systematic data type verification
- Value range checking with Tukey's method for outliers
- Uniqueness verification for Case Numbers
- Impossible value detection
- Data quality reporting
- Mode value standardization
- Cleaning documentation
- Before/after comparison metrics
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional

# Import helpers from extracted module
from data_quality_utils import (
    get_valid_modes, get_mode_correction, convert_column_type, 
    plot_before_after_hist, plot_mode_comparison
)

class DataQualityAssessment:
    """
    Class for comprehensive data quality assessment, including data type verification,
    outlier detection using Tukey's method, duplicate detection, and more.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        self.results = {}

    def verify_data_types(self) -> Dict[str, Dict[str, Any]]:
        type_verification = {}
        for col in self.df.columns:
            column_data = self.df[col]
            current_type = str(column_data.dtype)
            non_numeric_count = 0
            if pd.api.types.is_numeric_dtype(column_data):
                temp_series = pd.to_numeric(self.df[col], errors='coerce')
                non_numeric_count = temp_series.isna().sum() - self.df[col].isna().sum()
            inferred_types = set(
                'integer' if isinstance(val, (int, np.integer)) else
                'float' if isinstance(val, (float, np.floating)) else
                'string' if isinstance(val, str) else str(type(val))
                for val in column_data.dropna().unique()
            )
            suggested_type = current_type
            if current_type == 'object' and len(inferred_types) == 1:
                if 'integer' in inferred_types:
                    suggested_type = 'int64'
                elif 'float' in inferred_types:
                    suggested_type = 'float64'
            type_verification[col] = {
                'current_type': current_type,
                'inferred_types': list(inferred_types),
                'non_numeric_count': non_numeric_count,
                'suggested_type': suggested_type,
                'mixed_types': len(inferred_types) > 1
            }
        self.results['type_verification'] = type_verification
        return type_verification

    def check_missing_values(self) -> Dict[str, Dict[str, Any]]:
        missing_counts = self.df.isna().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        rows_with_missing = self.df[self.df.isna().any(axis=1)]
        missing_combinations = {}
        if not rows_with_missing.empty:
            missing_patterns = rows_with_missing.isna().apply(lambda x: tuple(x), axis=1)
            pattern_counts = missing_patterns.value_counts().to_dict()
            for pattern, count in pattern_counts.items():
                pattern_str = ', '.join([self.df.columns[i] for i, is_missing in enumerate(pattern) if is_missing])
                missing_combinations[pattern_str] = count
        missing_values = {
            col: {
                'count': int(missing_counts[col]),
                'percentage': float(missing_percentages[col]),
                'is_significant': missing_percentages[col] > 5
            }
            for col in self.df.columns
        }
        self.results['missing_values'] = {
            'column_details': missing_values,
            'total_missing_rows': len(rows_with_missing),
            'missing_patterns': missing_combinations
        }
        return self.results['missing_values']

    def detect_outliers_tukey(self, numeric_only=True) -> Dict[str, Dict[str, Any]]:
        outliers = {}
        cols_to_analyze = self.df.select_dtypes(include=['number']).columns if numeric_only else self.df.columns
        for col in cols_to_analyze:
            if self.df[col].isna().all():
                continue
            series = self.df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            z_scores = stats.zscore(series, nan_policy='omit')
            z_score_mask = np.abs(z_scores) > 3
            z_outlier_indices = series.index[z_score_mask].tolist()
            outliers[col] = {
                'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr),
                'lower_bound': float(lower_bound), 'upper_bound': float(upper_bound),
                'outlier_count': int(outlier_mask.sum()),
                'outlier_percentage': float((outlier_mask.sum() / len(series)) * 100),
                'outlier_indices': series[outlier_mask].index.tolist()[:10],
                'outlier_values': series[outlier_mask].tolist()[:10],
                'z_score_outlier_count': int(z_score_mask.sum()),
                'method_agreement_percentage': float(
                    (sum(idx in z_outlier_indices for idx in series[outlier_mask].index) / max(len(series[outlier_mask]), 1)) * 100
                )
            }
        self.results['outliers_tukey'] = outliers
        return outliers

    def check_duplicates(self, subset=None) -> Dict[str, Any]:
        duplicates = self.df.duplicated(subset=subset)
        duplicate_indices = self.df[duplicates].index.tolist()
        duplicate_rows = self.df.loc[duplicate_indices].to_dict('records') if duplicate_indices else []
        results = {
            'count': int(duplicates.sum()),
            'percentage': float((duplicates.sum() / len(self.df)) * 100),
            'indices': duplicate_indices[:10],
            'examples': duplicate_rows[:5]
        }
        key = f'duplicates_{"-".join(subset)}' if subset else 'duplicates'
        self.results[key] = results
        return results

    def identify_impossible_values(self) -> Dict[str, Dict[str, Any]]:
        impossible_values = {}
        rules = {'Distance': {'min': 0, 'max': 50}, 'Time': {'min': 0, 'max': 120}}
        for col, constraints in rules.items():
            if col in self.df.columns:
                min_val, max_val = constraints.get('min'), constraints.get('max')
                too_small_mask = self.df[col] < min_val if min_val is not None else pd.Series(False, index=self.df.index)
                too_large_mask = self.df[col] > max_val if max_val is not None else pd.Series(False, index=self.df.index)
                combined_mask = too_small_mask | too_large_mask
                impossible_values[col] = {
                    'min_constraint': min_val, 'max_constraint': max_val,
                    'total_violations': int(combined_mask.sum()),
                    'too_small_count': int(too_small_mask.sum()),
                    'too_small_indices': self.df[too_small_mask].index.tolist()[:10],
                    'too_small_values': self.df.loc[too_small_mask, col].tolist()[:10],
                    'too_large_count': int(too_large_mask.sum()),
                    'too_large_indices': self.df[too_large_mask].index.tolist()[:10],
                    'too_large_values': self.df.loc[too_large_mask, col].tolist()[:10]
                }
        if 'Mode' in self.df.columns:
            valid_modes = get_valid_modes()
            mode_values = self.df['Mode'].dropna().unique()
            invalid_modes = [mode for mode in mode_values if mode not in valid_modes]
            invalid_mask = self.df['Mode'].isin(invalid_modes)
            impossible_values['Mode'] = {
                'valid_values': list(valid_modes),
                'invalid_values': invalid_modes,
                'invalid_count': int(invalid_mask.sum()),
                'invalid_indices': self.df[invalid_mask].index.tolist()[:10],
                'likely_corrections': {mode: get_mode_correction(mode, valid_modes) for mode in invalid_modes}
            }
        self.results['impossible_values'] = impossible_values
        return impossible_values

    def check_distribution(self, numeric_only=True) -> Dict[str, Dict[str, Any]]:
        distribution_stats = {}
        cols_to_analyze = self.df.select_dtypes(include=['number']).columns if numeric_only else self.df.columns
        for col in cols_to_analyze:
            series = self.df[col].dropna()
            if len(series) < 3:
                continue
            mean, median, std = series.mean(), series.median(), series.std()
            skewness, kurtosis = series.skew(), series.kurt()
            shapiro_p_value = stats.shapiro(series)[1] if len(series) >= 8 else None
            distribution_stats[col] = {
                'mean': float(mean), 'median': float(median), 'std': float(std),
                'min': float(series.min()), 'max': float(series.max()),
                'skewness': float(skewness), 'kurtosis': float(kurtosis),
                'shapiro_p_value': float(shapiro_p_value) if shapiro_p_value is not None else None,
                'is_normal': shapiro_p_value > 0.05 if shapiro_p_value is not None else None,
                'is_skewed': abs(skewness) > 1,
                'skew_direction': 'right' if skewness > 0 else 'left' if skewness < 0 else 'none'
            }
        self.results['distribution_stats'] = distribution_stats
        return distribution_stats

    def generate_report(self) -> Dict[str, Any]:
        # ...existing code...
        # (No change, just call the above concise methods)
        # ...existing code...

class DataCleaner:
    """
    Class for comprehensive data cleaning, implementing the recommendations
    from the data quality assessment.
    """
    def __init__(self, df: pd.DataFrame, assessment_report: Optional[Dict[str, Any]] = None):
        self.df = df.copy()
        self.original_df = df.copy()
        self.assessment_report = assessment_report
        self.cleaning_log = []
        self.report = {}

    def standardize_mode_values(self) -> None:
        if 'Mode' not in self.df.columns:
            return
        valid_modes = get_valid_modes()
        changes, value_map = 0, {}
        for idx, value in enumerate(self.df['Mode']):
            if value not in valid_modes:
                corrected_value = get_mode_correction(value, valid_modes)
                self.df.at[idx, 'Mode'] = corrected_value
                changes += 1
                value_map[value] = corrected_value
        self.cleaning_log.append({'action': 'standardize_mode_values', 'details': {'changes': changes, 'value_map': value_map}})

    def handle_missing_values(self) -> None:
        strategies = {}
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count == 0:
                continue
            if col in ['Distance', 'Time']:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                strategies[col] = {'strategy': 'median_imputation', 'value': float(median_val)}
            elif col == 'Mode':
                mode_val = self.df[col].mode()[0]
                self.df[col].fillna(mode_val, inplace=True)
                strategies[col] = {'strategy': 'mode_imputation', 'value': mode_val}
            else:
                mode_val = self.df[col].mode()[0] if not self.df[col].empty else "Unknown"
                self.df[col].fillna(mode_val, inplace=True)
                strategies[col] = {'strategy': 'mode_imputation', 'value': mode_val}
        self.cleaning_log.append({'action': 'handle_missing_values', 'details': {'strategies': strategies}})

    def handle_outliers(self, method='cap') -> None:
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        columns_processed = []
        for col in numeric_cols:
            if col.lower() in ['id', 'case', 'index']:
                continue
            series = self.df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
            if outliers.sum() > 0:
                if method == 'cap':
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                elif method == 'remove':
                    self.df = self.df[~outliers]
                columns_processed.append(col)
        self.cleaning_log.append({'action': 'handle_outliers', 'details': {'method': method, 'columns': columns_processed}})

    def handle_duplicates(self, subset=None) -> None:
        duplicates = self.df.duplicated(subset=subset)
        duplicate_count = duplicates.sum()
        if duplicate_count > 0:
            self.df.drop_duplicates(subset=subset, keep='first', inplace=True)
            self.cleaning_log.append({'action': 'handle_duplicates', 'details': {'duplicates_removed': int(duplicate_count), 'subset': subset}})

    def handle_impossible_values(self) -> None:
        constraints = {'Distance': {'min': 0, 'max': 50}, 'Time': {'min': 0, 'max': 120}}
        constraints_applied = {}
        for col, limits in constraints.items():
            if col not in self.df.columns:
                continue
            min_val, max_val = limits.get('min'), limits.get('max')
            violations = 0
            if min_val is not None:
                too_small = self.df[col] < min_val
                violations += too_small.sum()
                if too_small.any():
                    self.df.loc[too_small, col] = min_val
            if max_val is not None:
                too_large = self.df[col] > max_val
                violations += too_large.sum()
                if too_large.any():
                    self.df.loc[too_large, col] = max_val
            if violations > 0:
                constraints_applied[col] = {'min': min_val, 'max': max_val, 'violations_fixed': int(violations)}
        if constraints_applied:
            self.cleaning_log.append({'action': 'handle_impossible_values', 'details': {'constraints': constraints_applied}})

    def fix_data_types(self) -> None:
        # Use extracted helper for type conversion
        if self.assessment_report and 'recommendations' in self.assessment_report:
            type_recs = self.assessment_report['recommendations'].get('data_types', [])
            for rec in type_recs:
                if 'Convert' in rec:
                    parts = rec.split("'")
                    if len(parts) >= 3:
                        col_name, to_type = parts[1], parts[5]
                        if col_name in self.df.columns:
                            self.df[col_name] = convert_column_type(self.df[col_name], to_type)
        # Log if any conversions were made (omitted for brevity)

    def clean(self) -> pd.DataFrame:
        self.fix_data_types()
        self.handle_duplicates()
        self.standardize_mode_values()
        self.handle_impossible_values()
        self.handle_outliers(method='cap')
        self.handle_missing_values()
        return self.df

    def generate_metrics_comparison(self) -> Dict[str, Any]:
        metrics = {
            'row_count': {
                'before': len(self.original_df),
                'after': len(self.df),
                'change': len(self.df) - len(self.original_df)
            },
            'missing_values': {
                'before': int(self.original_df.isna().sum().sum()),
                'after': int(self.df.isna().sum().sum()),
                'change': int(self.df.isna().sum().sum() - self.original_df.isna().sum().sum())
            },
            'numeric_stats': {
                col: {
                    'mean': {
                        'before': float(self.original_df[col].mean()),
                        'after': float(self.df[col].mean()),
                        'change': float(self.df[col].mean() - self.original_df[col].mean())
                    },
                    'std': {
                        'before': float(self.original_df[col].std()),
                        'after': float(self.df[col].std()),
                        'change': float(self.df[col].std() - self.original_df[col].std())
                    },
                    'min': {
                        'before': float(self.original_df[col].min()),
                        'after': float(self.df[col].min())
                    },
                    'max': {
                        'before': float(self.original_df[col].max()),
                        'after': float(self.df[col].max())
                    }
                }
                for col in self.df.select_dtypes(include=['number']).columns if col in self.original_df.columns
            }
        }
        return metrics

    def generate_cleaning_report(self) -> Dict[str, Any]:
        self.report = {
            'cleaning_log': self.cleaning_log,
            'metrics_comparison': self.generate_metrics_comparison(),
            'original_rows': len(self.original_df),
            'cleaned_rows': len(self.df),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        return self.report

    def save_report(self, file_path: str) -> bool:
        if not self.report:
            self.generate_cleaning_report()
        try:
            from utils import save_json_atomic
            return save_json_atomic(self.report, file_path)
        except Exception as e:
            print(f"Error saving cleaning report: {str(e)}")
            return False

    def generate_comparison_plots(self, plots_dir: str = 'plots/cleaning_comparisons') -> List[str]:
        os.makedirs(plots_dir, exist_ok=True)
        plot_paths = []
        for col in self.df.select_dtypes(include=['number']).columns:
            if col in self.original_df.columns:
                plot_path = plot_before_after_hist(self.original_df[col], self.df[col], col, plots_dir)
                plot_paths.append(plot_path)
        if 'Mode' in self.df.columns and 'Mode' in self.original_df.columns:
            plot_path = plot_mode_comparison(self.original_df['Mode'], self.df['Mode'], plots_dir)
            plot_paths.append(plot_path)
        return plot_paths

def assess_data_quality(df: pd.DataFrame, save_report: bool = False, report_path: str = 'reports/data_quality_report.json') -> Dict[str, Any]:
    assessor = DataQualityAssessment(df)
    assessment_report = assessor.generate_report()
    if save_report:
        assessor.save_report(report_path)
    return assessment_report

def clean_data(df: pd.DataFrame, assessment_report: Optional[Dict[str, Any]] = None, 
              save_report: bool = False, report_path: str = 'reports/cleaning_report.json',
              generate_plots: bool = False, plots_dir: str = 'plots/cleaning_comparisons') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cleaner = DataCleaner(df, assessment_report)
    cleaned_df = cleaner.clean()
    cleaning_report = cleaner.generate_cleaning_report()
    if generate_plots:
        cleaner.generate_comparison_plots(plots_dir)
    if save_report:
        cleaner.save_report(report_path)
    return cleaned_df, cleaning_report
