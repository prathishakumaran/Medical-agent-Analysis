from langchain.tools import tool

@tool
def recommend_specialist(conditions: str) -> str:
    """
    Recommend a medical specialist based on a patient's conditions or symptoms.
    Input should be a comma-separated list of conditions (e.g., 'high blood pressure, diabetes').
    """
    conditions = conditions.lower()
    recommendations = []
    
    # Mapping conditions to specialists
    specialty_map = {
        'cardiologist': ['heart', 'blood pressure', 'hypertension', 'chest pain', 'cholesterol', 'bp'],
        'endocrinologist': ['diabetes', 'thyroid', 'glucose', 'hormone', 'sugar'],
        'pulmonologist': ['asthma', 'breathing', 'lungs', 'copd', 'cough'],
        'neurologist': ['migraine', 'headache', 'seizure', 'dizziness', 'nerve'],
        'gastroenterologist': ['stomach', 'digestion', 'ulcer', 'reflux', 'liver'],
        'rheumatologist': ['arthritis', 'joint pain', 'lupus', 'inflammation'],
        'nephrologist': ['kidney', 'renal']
    }
    
    for specialty, keywords in specialty_map.items():
        for keyword in keywords:
            if keyword in conditions and specialty.title() not in recommendations:
                recommendations.append(specialty.title())
                
    if not recommendations:
        return "Based on the provided conditions, a General Practitioner (GP) or Internal Medicine physician is recommended for initial evaluation."
        
    response = f"Based on the conditions ({conditions}), the recommended specialist(s) are: "
    response += ", ".join(recommendations)
    return response
