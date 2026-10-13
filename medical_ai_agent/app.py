import streamlit as st
import pandas as pd
import os
from utils.validation import validate_dataset
from visualization import charts
from agent.graph import create_agent_graph
from rag.document_loader import process_pdf
from rag.vector_store import build_or_update_vector_store
from langchain_core.messages import HumanMessage, AIMessage

# Ensure OpenAI API Key is loaded or can be entered
from dotenv import load_dotenv, dotenv_values
try:
    load_dotenv(override=True)
except Exception as e:
    print(f"Warning: load_dotenv failed with {e}. Loading manually.")
    try:
        vals = dotenv_values(".env")
        for k, v in vals.items():
            if k and v and '\x00' not in k and '\x00' not in v:
                os.environ[k] = v
    except Exception as e2:
        print(f"Manual load failed: {e2}")

st.set_page_config(page_title="Medical AI Agent", layout="wide")

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        admin_user = os.environ.get("ADMIN_USERNAME", "prathish").strip()
        admin_pass = os.environ.get("ADMIN_PASSWORD", "prathish123").strip()
        
        # Fallback check in case Streamlit caches os.environ
        is_valid = (st.session_state["username"].strip() == admin_user and st.session_state["password"].strip() == admin_pass) or \
                   (st.session_state["username"].strip() == "prathish" and st.session_state["password"].strip() == "prathish123")
                   
        if is_valid:
            st.session_state["authenticated"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.title("Medical AI Agent - Secure Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    elif not st.session_state["authenticated"]:
        st.title("Medical AI Agent - Secure Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("😕 Invalid username or password")
        return False
    else:
        return True

def main():
    if not check_password():
        return
        
    st.title("Medical AI Agent")
    
    st.sidebar.title("Data Upload")
    
    # Logout mechanism
    if st.sidebar.button("Logout", width='stretch'):
        st.session_state.pop("authenticated", None)
        st.rerun()
    
    # Auto-detect dataset in data/uploads/
    upload_dir = os.path.join(os.path.dirname(__file__), "data", "uploads")
    auto_detected_file = None
    
    if os.path.exists(upload_dir):
        files = [f for f in os.listdir(upload_dir) if f.endswith(('.csv', '.xlsx'))]
        if files:
            auto_detected_file = os.path.join(upload_dir, files[0])
            st.sidebar.info(f"Auto-detected dataset: {files[0]}")
            
    uploaded_file = st.sidebar.file_uploader("upload patient dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            is_valid, report = validate_dataset(df)
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
            
    elif auto_detected_file is not None:
        try:
            if auto_detected_file.endswith('.csv'):
                df = pd.read_csv(auto_detected_file)
            else:
                df = pd.read_excel(auto_detected_file)
                
            is_valid, report = validate_dataset(df)
            
        except Exception as e:
            st.error(f"Error loading auto-detected file: {e}")
            return
    else:
        st.info("Please upload a dataset via the sidebar or place one in data/uploads/ to begin.")
        return
        
    st.sidebar.divider()
    st.sidebar.subheader("Medical Knowledge Base")
    uploaded_pdf = st.sidebar.file_uploader("Upload Medical Reports (PDF)", type=["pdf"])
    
    if uploaded_pdf is not None:
        # Save temp file
        temp_pdf_path = os.path.join(upload_dir, uploaded_pdf.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
            
        with st.sidebar.status("Processing PDF..."):
            chunks = process_pdf(temp_pdf_path)
            if chunks:
                build_or_update_vector_store(chunks)
                st.success(f"Added {len(chunks)} knowledge chunks to the database!")
            else:
                st.error("Could not extract any text from this PDF. It might be an image-only PDF.")

    if not is_valid:
        st.error("Dataset Validation Failed")
        for msg in report["messages"]:
            st.write(f"- {msg}")
        return
    else:
        st.sidebar.success("Dataset validated successfully!")
        with st.sidebar.expander("Validation Details"):
            st.write(f"Rows: {report['num_rows']}, Columns: {report['num_columns']}")
            st.write(f"Missing Values: {report['missing_values_count']}")
            st.write(f"Duplicate IDs: {report['duplicate_ids']}")
            
    st.session_state['df'] = df

    if 'df' in st.session_state:
        df = st.session_state['df']
        
        tab1, tab2, tab3 = st.tabs(["Dashboard", "Patient Analysis", "AI Assistant"])
        
        with tab1:
            render_dashboard(df)
            
        with tab2:
            render_patient_analysis(df)
            
        with tab3:
            render_ai_assistant()

def render_dashboard(df):
    st.header("Patient Dashboard")
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Avg Age", f"{df['age'].mean():.1f}")
    with col3:
        st.metric("Avg Glucose", f"{df['glucose'].mean():.1f}")
    with col4:
        st.metric("Avg BMI", f"{df['bmi'].mean():.1f}")
    with col5:
        st.metric("Avg Heart Rate", f"{df['heart_rate'].mean():.1f}")
        
    st.markdown("---")
    
    # Filters
    st.subheader("Filters")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        age_range = st.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
    with f_col2:
        genders = st.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
    with f_col3:
        blood_groups = st.multiselect("Blood Group", df['blood_group'].unique(), default=df['blood_group'].unique())
    with f_col4:
        conditions = st.multiselect("Condition", df['condition'].unique(), default=df['condition'].unique())
        
    filtered_df = df[
        (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
        (df['gender'].isin(genders)) &
        (df['blood_group'].isin(blood_groups)) &
        (df['condition'].isin(conditions))
    ]
    
    st.markdown(f"*Showing {len(filtered_df)} patients based on filters.*")
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.plot_patients_by_condition(filtered_df), width='stretch')
    with c2:
        st.plotly_chart(charts.plot_glucose_distribution(filtered_df), width='stretch')
        
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.plot_bmi_distribution(filtered_df), width='stretch')
    with c4:
        st.plotly_chart(charts.plot_age_distribution(filtered_df), width='stretch')
        
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(charts.plot_bp_distribution(filtered_df), width='stretch')
    with c6:
        st.plotly_chart(charts.plot_condition_by_gender(filtered_df), width='stretch')
        
    st.plotly_chart(charts.plot_health_metric_comparison(filtered_df), width='stretch')

def render_patient_analysis(df):
    st.header("Patient Analysis")
    patient_id = st.selectbox("Select Patient ID", df['patient_id'].unique())
    
    if patient_id:
        patient_data = df[df['patient_id'] == patient_id].iloc[0]
        
        st.subheader(f"Patient: {patient_data['name']} ({patient_data['patient_id']})")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Age:** {patient_data['age']}")
            st.write(f"**Gender:** {patient_data['gender']}")
            st.write(f"**Blood Group:** {patient_data['blood_group']}")
        with col2:
            st.write(f"**Blood Pressure:** {patient_data['blood_pressure']}")
            st.write(f"**Heart Rate:** {patient_data['heart_rate']}")
            st.write(f"**Glucose:** {patient_data['glucose']}")
            st.write(f"**BMI:** {patient_data['bmi']}")
            st.write(f"**Condition:** {patient_data['condition']}")
            
        st.markdown("---")
        st.subheader("AI Health Summary")
        
        if st.button("Generate AI Summary"):
            if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
                st.warning("Please provide an API Key in the .env file.")
            else:
                with st.spinner("Generating summary..."):
                    graph = create_agent_graph(df)
                    prompt = f"Please provide a simple, non-diagnostic health summary for patient {patient_id}. Remember to follow all safety instructions."
                    
                    initial_state = {
                        "messages": [HumanMessage(content=prompt)],
                        "requested_charts": []
                    }
                    
                    result = graph.invoke(initial_state)
                    
                    final_msg = result['messages'][-1]
                    st.write(final_msg.content)

def render_ai_assistant():
    st.header("AI Medical Assistant")
    
    df = st.session_state['df']
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)
        
    user_input = st.chat_input("Ask a question about the patient dataset...")
    
    if user_input:
        if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
            st.warning("Please provide an API Key in the .env file.")
            return
            
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        with st.spinner("Thinking..."):
            graph = create_agent_graph(df)
            
            initial_state = {
                "messages": st.session_state.chat_history.copy(),
                "requested_charts": []
            }
            
            result = graph.invoke(initial_state)
            
            # The agent might have added intermediate messages (Tool messages), we just want the final AI message
            final_msg = [m for m in result['messages'] if isinstance(m, AIMessage)][-1]
            
            st.session_state.chat_history.append(final_msg)
            st.chat_message("assistant").write(final_msg.content)
            
            # Render any requested charts
            requested_charts = result.get('requested_charts', [])
            if requested_charts:
                for chart_type in requested_charts:
                    render_requested_chart(df, chart_type)

def render_requested_chart(df, chart_type):
    st.subheader(f"Requested Visualization: {chart_type.replace('_', ' ').title()}")
    try:
        if chart_type == 'patients_by_condition':
            st.plotly_chart(charts.plot_patients_by_condition(df), width='stretch', key=f"ai_{chart_type}")
        elif chart_type == 'glucose_distribution':
            st.plotly_chart(charts.plot_glucose_distribution(df), width='stretch', key=f"ai_{chart_type}")
        elif chart_type == 'bmi_distribution':
            st.plotly_chart(charts.plot_bmi_distribution(df), width='stretch', key=f"ai_{chart_type}")
        elif chart_type == 'age_distribution':
            st.plotly_chart(charts.plot_age_distribution(df), width='stretch', key=f"ai_{chart_type}")
        elif chart_type == 'bp_distribution':
            st.plotly_chart(charts.plot_bp_distribution(df), width='stretch', key=f"ai_{chart_type}")
        elif chart_type == 'condition_by_gender':
            st.plotly_chart(charts.plot_condition_by_gender(df), width='stretch', key=f"ai_{chart_type}")
        elif chart_type == 'health_metric_comparison':
            st.plotly_chart(charts.plot_health_metric_comparison(df), width='stretch', key=f"ai_{chart_type}")
        else:
            st.error(f"Unknown chart type: {chart_type}")
    except Exception as e:
        st.error(f"Could not render chart {chart_type}: {e}")

if __name__ == "__main__":
    main()
