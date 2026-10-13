from langchain_core.tools import tool
import pandas as pd

def get_patient_tools(df: pd.DataFrame):
    
    @tool
    def get_patient_info(patient_id: str) -> str:
        """Retrieves detailed health information for a specific patient by their ID (e.g., 'P0001')."""
        patient = df[df['patient_id'] == patient_id]
        if patient.empty:
            return f"No patient found with ID: {patient_id}"
        
        p = patient.iloc[0]
        return (
            f"Patient {p['patient_id']} ({p['name']}): "
            f"Age {p['age']}, Gender {p['gender']}, Blood Group {p['blood_group']}, "
            f"BP {p['blood_pressure']}, Heart Rate {p['heart_rate']} bpm, "
            f"Glucose {p['glucose']} mg/dL, BMI {p['bmi']}, Condition: {p['condition']}"
        )
        
    return [get_patient_info]
