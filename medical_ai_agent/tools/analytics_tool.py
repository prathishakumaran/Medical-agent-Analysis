from langchain_core.tools import tool
import pandas as pd
from typing import Optional

def get_analytics_tools(df: pd.DataFrame):
    
    @tool
    def calculate_average(metric: str) -> str:
        """Calculates the average of a specified metric (e.g., 'glucose', 'bmi', 'age', 'heart_rate')."""
        if metric.lower() not in df.columns:
            return f"Metric '{metric}' not found in dataset."
        try:
            avg = df[metric.lower()].mean()
            return f"The average {metric} is {avg:.2f}."
        except Exception as e:
            return f"Could not calculate average for {metric}: {e}"

    @tool
    def count_patients(condition: Optional[str] = None) -> str:
        """Counts the total number of patients, optionally filtered by condition."""
        if condition:
            count = len(df[df['condition'].str.lower() == condition.lower()])
            return f"There are {count} patients with condition '{condition}'."
        return f"There are a total of {len(df)} patients."

    @tool
    def count_abnormal_values() -> str:
        """Counts patients with abnormal health metrics (BMI > 30 or Glucose > 100)."""
        high_bmi = len(df[df['bmi'] > 30])
        high_glucose = len(df[df['glucose'] > 100])
        return f"There are {high_bmi} patients with BMI > 30 and {high_glucose} patients with Glucose > 100."

    return [calculate_average, count_patients, count_abnormal_values]
