from langchain_core.tools import tool

def get_visualization_tools():
    
    @tool
    def request_chart(chart_type: str) -> str:
        """
        Requests the UI to render a specific chart.
        Allowed chart_type values:
        - 'patients_by_condition'
        - 'glucose_distribution'
        - 'bmi_distribution'
        - 'age_distribution'
        - 'bp_distribution'
        - 'condition_by_gender'
        - 'health_metric_comparison'
        """
        valid_charts = [
            'patients_by_condition', 'glucose_distribution', 'bmi_distribution',
            'age_distribution', 'bp_distribution', 'condition_by_gender',
            'health_metric_comparison'
        ]
        
        if chart_type in valid_charts:
            # The agent will include this exact string in its thought process/tool output
            # We will intercept this in Streamlit to actually render the chart.
            return f"CHART_REQUESTED:{chart_type}"
        else:
            return f"Invalid chart type requested. Available options: {', '.join(valid_charts)}"
            
    return [request_chart]
