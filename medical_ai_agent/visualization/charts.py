import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_patients_by_condition(df: pd.DataFrame):
    """Plot patients by condition."""
    counts = df['condition'].value_counts().reset_index()
    counts.columns = ['Condition', 'Count']
    fig = px.bar(counts, x='Condition', y='Count', title="Patients by Condition", 
                 color='Condition', text='Count')
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)
    return fig

def plot_glucose_distribution(df: pd.DataFrame):
    """Plot glucose distribution."""
    fig = px.histogram(df, x='glucose', nbins=20, title="Glucose Level Distribution", 
                       color_discrete_sequence=['indianred'])
    fig.update_layout(xaxis_title="Glucose (mg/dL)", yaxis_title="Count")
    return fig

def plot_bmi_distribution(df: pd.DataFrame):
    """Plot BMI distribution."""
    fig = px.histogram(df, x='bmi', nbins=20, title="BMI Distribution", 
                       color_discrete_sequence=['teal'])
    fig.update_layout(xaxis_title="BMI", yaxis_title="Count")
    return fig

def plot_age_distribution(df: pd.DataFrame):
    """Plot age distribution."""
    fig = px.histogram(df, x='age', nbins=20, title="Age Distribution",
                       color_discrete_sequence=['cornflowerblue'])
    fig.update_layout(xaxis_title="Age", yaxis_title="Count")
    return fig

def plot_bp_distribution(df: pd.DataFrame):
    """
    Plot blood pressure distribution.
    Assumes format is 'systolic/diastolic'.
    """
    # Create a copy to not modify the original df
    df_copy = df.copy()
    if 'blood_pressure' in df_copy.columns:
        # Split into systolic and diastolic
        try:
            df_copy[['Systolic', 'Diastolic']] = df_copy['blood_pressure'].str.split('/', expand=True).astype(float)
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=df_copy['Systolic'], name="Systolic"))
            fig.add_trace(go.Box(y=df_copy['Diastolic'], name="Diastolic"))
            fig.update_layout(title="Blood Pressure Distribution", yaxis_title="mmHg")
            return fig
        except Exception as e:
            # If parsing fails, just return a blank figure with an annotation
            fig = go.Figure()
            fig.add_annotation(text="Could not parse blood pressure values.", showarrow=False)
            return fig
    return go.Figure()

def plot_condition_by_gender(df: pd.DataFrame):
    """Plot condition by gender."""
    counts = df.groupby(['condition', 'gender']).size().reset_index(name='Count')
    fig = px.bar(counts, x='condition', y='Count', color='gender', barmode='group',
                 title="Condition by Gender")
    fig.update_layout(xaxis_title="Condition")
    return fig

def plot_health_metric_comparison(df: pd.DataFrame):
    """Plot a comparison of health metrics (Glucose vs BMI colored by Condition)."""
    fig = px.scatter(df, x='bmi', y='glucose', color='condition', hover_data=['patient_id', 'age'],
                     title="Glucose vs BMI by Condition")
    fig.update_layout(xaxis_title="BMI", yaxis_title="Glucose (mg/dL)")
    return fig
