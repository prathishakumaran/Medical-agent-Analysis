import pandas as pd
from typing import Tuple, Dict, Any, List

REQUIRED_COLUMNS = [
    'patient_id', 'name', 'age', 'gender', 'blood_group',
    'blood_pressure', 'heart_rate', 'glucose', 'bmi', 'condition'
]

NUMERICAL_COLUMNS = ['age', 'heart_rate', 'glucose', 'bmi']

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates the uploaded patient dataset.
    Returns a tuple: (is_valid, validation_report)
    """
    report = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_columns": [],
        "missing_values_count": 0,
        "duplicate_ids": 0,
        "invalid_numerical": [],
        "is_valid": True,
        "messages": []
    }

    # 1. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        report["missing_columns"] = missing_cols
        report["is_valid"] = False
        report["messages"].append(f"Missing required columns: {', '.join(missing_cols)}")
        return False, report # Fatal error, stop here

    # 2. Check missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    report["missing_values_count"] = int(total_missing)
    if total_missing > 0:
        report["messages"].append(f"Found {total_missing} missing values across the dataset.")
        # We might not invalidate it entirely, but warn

    # 3. Check duplicate IDs
    if 'patient_id' in df.columns:
        duplicates = df.duplicated(subset=['patient_id']).sum()
        report["duplicate_ids"] = int(duplicates)
        if duplicates > 0:
            report["is_valid"] = False
            report["messages"].append(f"Found {duplicates} duplicate patient IDs. Please ensure IDs are unique.")

    # 4. Validate numerical columns
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            # Check if column can be converted to numeric
            non_numeric = pd.to_numeric(df[col], errors='coerce').isnull().sum() - df[col].isnull().sum()
            if non_numeric > 0:
                report["invalid_numerical"].append(col)
                report["is_valid"] = False
                report["messages"].append(f"Column '{col}' contains non-numerical values.")
            
    if report["is_valid"]:
        report["messages"].insert(0, "Dataset validation successful!")
        
    return report["is_valid"], report
