import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import base64
import plotly.express as px
from fpdf import FPDF
from setup import download_data
from utils.preprocessing import load_data, get_data_summary, clean_and_preprocess
from utils.clustering import find_optimal_k, train_kmeans, assign_personas, generate_insights
from utils.visualization import (plot_distribution, plot_correlation_heatmap, 
                                 plot_pairplot, plot_scatter, plot_elbow_curve, 
                                 plot_silhouette_scores, plot_3d_clusters, 
                                 plot_cluster_radar, plot_cluster_distribution)
from utils.prediction import load_models, predict_customer_segment

# --- Page Config ---
st.set_page_config(page_title="AI-Powered Customer Segmentation & Business Intelligence Dashboard", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        color: #A0A0A0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #4DB6AC;
        font-size: 32px;
        font-weight: bold;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Ensure Data exists ---
if not os.path.exists("data/Mall_Customers.csv"):
    download_data()

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'kmeans' not in st.session_state:
    scaler, kmeans = load_models()
    st.session_state.kmeans = kmeans
    st.session_state.scaler = scaler
if 'optimal_k' not in st.session_state:
    st.session_state.optimal_k = 5
if 'personas_map' not in st.session_state:
    st.session_state.personas_map = None
if 'insights' not in st.session_state:
    st.session_state.insights = None

# --- Sidebar Navigation ---
st.sidebar.title("Navigation 🧭")
options = [
    "Dashboard", 
    "Dataset Overview", 
    "EDA", 
    "Data Preprocessing", 
    "Model Training", 
    "Cluster Analysis", 
    "Customer Prediction", 
    "Business Insights", 
    "Reports", 
    "About"
]
choice = st.sidebar.radio("Go to", options)

df = st.session_state.df

if df is None:
    st.error("Failed to load dataset. Please check the data folder.")
    st.stop()

# --- Helper function for Metric Cards ---
def create_metric_card(title, value):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
#                 DASHBOARD
# ==========================================
if choice == "Dashboard":
    st.title("🛒 AI-Powered Customer Segmentation & Business Intelligence Dashboard")
    st.markdown("Analyze customer behaviors and segments using AI & K-Means Clustering.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Total Customers", f"{len(df)}")
    with col2:
        create_metric_card("Optimal Clusters", f"{st.session_state.optimal_k}")
    with col3:
        avg_income = round(df['Annual Income (k$)'].mean(), 2)
        create_metric_card("Avg Income (k$)", f"${avg_income}")
    with col4:
        avg_spending = round(df['Spending Score (1-100)'].mean(), 2)
        create_metric_card("Avg Spending", f"{avg_spending}/100")
        
    st.markdown("---")
    
    if st.session_state.df_clean is not None and 'Persona' in st.session_state.df_clean.columns:
        st.subheader("Cluster Distribution")
        fig = plot_cluster_distribution(st.session_state.df_clean)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Train the model and assign clusters in 'Model Training' to see cluster distributions here.")

# ==========================================
#            DATASET OVERVIEW
# ==========================================
elif choice == "Dataset Overview":
    st.title("📊 Dataset Overview")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    summary = get_data_summary(df)
    
    with col1:
        st.markdown("**Dataset Shape:**")
        st.write(f"{summary['shape'][0]} rows, {summary['shape'][1]} columns")
        
        st.markdown("**Missing Values:**")
        st.write(pd.Series(summary['missing_values']))
        
    with col2:
        st.markdown("**Data Types:**")
        st.write(pd.Series(summary['dtypes']))
        
        st.markdown("**Duplicate Rows:**")
        st.write(summary['duplicates'])
        
    st.subheader("Summary Statistics")
    st.dataframe(pd.DataFrame(summary['describe']), use_container_width=True)
    
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Dataset as CSV",
        data=csv,
        file_name='Mall_Customers.csv',
        mime='text/csv',
    )

# ==========================================
#                   EDA
# ==========================================
elif choice == "EDA":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Interactive visualizations to understand data distributions and relationships.")
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Relationships", "Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_distribution(df, 'Age', 'Age Distribution', ['#636EFA']), use_container_width=True)
            st.plotly_chart(plot_distribution(df, 'Spending Score (1-100)', 'Spending Score Distribution', ['#AB63FA']), use_container_width=True)
        with col2:
            st.plotly_chart(plot_distribution(df, 'Annual Income (k$)', 'Income Distribution', ['#EF553B']), use_container_width=True)
            fig_gender = px.pie(df, names='Gender', title='Gender Distribution', template="plotly_dark", hole=0.3)
            st.plotly_chart(fig_gender, use_container_width=True)
            
    with tab2:
        st.plotly_chart(plot_pairplot(df), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_scatter(df, 'Age', 'Spending Score (1-100)', 'Gender', 'Age vs Spending Score'), use_container_width=True)
        with col2:
            st.plotly_chart(plot_scatter(df, 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender', 'Income vs Spending Score'), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

# ==========================================
#           DATA PREPROCESSING
# ==========================================
elif choice == "Data Preprocessing":
    st.title("⚙️ Data Preprocessing")
    
    if st.button("Run Preprocessing"):
        with st.spinner("Cleaning, Encoding, and Scaling data..."):
            time.sleep(1) # Simulate processing time
            df_clean, X_scaled, features = clean_and_preprocess(df)
            st.session_state.df_clean = df_clean
            st.session_state.X_scaled = X_scaled
            st.session_state.features = features
            st.session_state.scaler = load_models()[0] # Load scaler that was saved
            st.success("Preprocessing Complete!")
            
    if st.session_state.df_clean is not None:
        st.subheader("Processed Dataset")
        st.dataframe(st.session_state.df_clean.head(), use_container_width=True)
        
        st.subheader("Scaled Features (X)")
        st.dataframe(pd.DataFrame(st.session_state.X_scaled, columns=st.session_state.features).head(), use_container_width=True)
        
        st.info(f"Features selected for clustering: {', '.join(st.session_state.features)}")
    else:
        st.warning("Click 'Run Preprocessing' to prepare data for clustering.")

# ==========================================
#            MODEL TRAINING
# ==========================================
elif choice == "Model Training":
    st.title("🧠 Model Training (K-Means)")
    
    if st.session_state.X_scaled is None:
        st.warning("Please run Data Preprocessing first.")
    else:
        st.markdown("### 1. Determine Optimal K")
        if st.button("Evaluate K (Elbow & Silhouette)"):
            with st.spinner("Calculating metrics..."):
                metrics_df = find_optimal_k(st.session_state.X_scaled)
                st.session_state.metrics_df = metrics_df
            st.success("Evaluation complete.")
        
        if 'metrics_df' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_elbow_curve(st.session_state.metrics_df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_silhouette_scores(st.session_state.metrics_df), use_container_width=True)
                
            # Default optimal k is often 5 for this dataset
            st.session_state.optimal_k = st.slider("Select Optimal K based on charts", min_value=2, max_value=10, value=5)
            
            st.markdown("### 2. Train Model")
            if st.button(f"Train K-Means with K={st.session_state.optimal_k}"):
                with st.spinner("Training model..."):
                    kmeans = train_kmeans(st.session_state.X_scaled, st.session_state.optimal_k)
                    st.session_state.kmeans = kmeans
                    
                    # Predict and map
                    df_clean = st.session_state.df_clean
                    df_clean['Cluster'] = kmeans.labels_
                    
                    df_clean, personas_map = assign_personas(df_clean, 'Cluster')
                    st.session_state.df_clean = df_clean
                    st.session_state.personas_map = personas_map
                    
                    st.session_state.insights = generate_insights(personas_map)
                    
                st.success("Model trained and saved successfully!")
                st.info("Head over to 'Cluster Analysis' to view the results.")

# ==========================================
#           CLUSTER ANALYSIS
# ==========================================
elif choice == "Cluster Analysis":
    st.title("🧩 Cluster Analysis")
    
    if st.session_state.kmeans is None or 'Persona' not in st.session_state.df_clean.columns:
        st.warning("Please train the model first in the 'Model Training' section.")
    else:
        df_clean = st.session_state.df_clean
        
        tab1, tab2, tab3 = st.tabs(["3D Scatter Plot", "Radar Chart", "2D Scatter (PCA Proxy)"])
        
        with tab1:
            st.markdown("Interactive 3D view of clusters based on Age, Income, and Spending.")
            st.plotly_chart(plot_3d_clusters(df_clean), use_container_width=True)
            
        with tab2:
            st.markdown("Profile of each cluster showing normalized feature averages.")
            st.plotly_chart(plot_cluster_radar(df_clean), use_container_width=True)
            
        with tab3:
            st.markdown("2D projection using Income and Spending Score.")
            st.plotly_chart(plot_scatter(df_clean, 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster', "Income vs Spending (Colored by Cluster)"), use_container_width=True)

        st.subheader("Cluster Mapping (Personas)")
        personas_df = pd.DataFrame(list(st.session_state.personas_map.items()), columns=['Cluster', 'Persona assigned'])
        st.table(personas_df)

# ==========================================
#          CUSTOMER PREDICTION
# ==========================================
elif choice == "Customer Prediction":
    st.title("🎯 Customer Prediction")
    st.markdown("Enter details for a new customer to predict their segment and get recommendations.")
    
    if st.session_state.kmeans is None:
        st.warning("Please train the model first.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age", min_value=15, max_value=100, value=30)
            with col2:
                income = st.number_input("Annual Income (k$)", min_value=10, max_value=200, value=50)
                spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
                
            submit = st.form_submit_button("Predict Segment")
            
        if submit:
            result = predict_customer_segment(
                age, income, spending, 
                st.session_state.scaler, 
                st.session_state.kmeans, 
                st.session_state.personas_map
            )
            
            st.markdown("### Prediction Result")
            c1, c2, c3 = st.columns(3)
            c1.metric("Assigned Cluster", f"Cluster {result['cluster']}")
            c2.metric("Customer Type", result['persona'])
            c3.metric("Confidence", f"{result['confidence']:.2f}%")
            
            st.info(f"**Business Recommendation:** {result['recommendation']}")

# ==========================================
#           BUSINESS INSIGHTS
# ==========================================
elif choice == "Business Insights":
    st.title("💡 Business Insights")
    
    if st.session_state.insights is None:
        st.warning("Please train the model first to generate insights.")
    else:
        st.markdown("### Automated Marketing Recommendations")
        for insight in st.session_state.insights:
            st.markdown(f"- {insight}")
            
        st.markdown("---")
        st.subheader("Cluster Summary Table")
        df_clean = st.session_state.df_clean
        summary_table = df_clean.groupby('Persona')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
        summary_table['Count'] = df_clean['Persona'].value_counts()
        st.dataframe(summary_table, use_container_width=True)

# ==========================================
#                 REPORTS
# ==========================================
elif choice == "Reports":
    st.title("📄 Export Reports")
    
    if st.session_state.df_clean is None or 'Persona' not in st.session_state.df_clean.columns:
        st.warning("Train the model and generate clusters before exporting reports.")
    else:
        st.markdown("Download generated insights and cluster summaries.")
        
        # CSV Export
        df_clean = st.session_state.df_clean
        csv = df_clean.to_csv(index=False)
        st.download_button("Download Clustered Data (CSV)", data=csv, file_name="Clustered_Customers.csv", mime="text/csv")
        
        # PDF Generation (Basic example using FPDF)
        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, "AI-Powered Customer Segmentation & Business Intelligence Report", ln=1, align="C")
            pdf.ln(10)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Business Insights:", ln=1)
            pdf.set_font("Arial", size=10)
            for insight in st.session_state.insights:
                # Remove markdown formatting for PDF
                clean_insight = insight.replace("**", "")
                pdf.multi_cell(0, 8, "- " + clean_insight)
                
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Cluster Statistics:", ln=1)
            pdf.set_font("Arial", size=10)
            
            summary_table = df_clean.groupby('Persona')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
            for persona, row in summary_table.iterrows():
                stat_str = f"{persona} - Age: {row['Age']}, Income: {row['Annual Income (k$)']}k, Spending: {row['Spending Score (1-100)']}"
                pdf.cell(200, 8, stat_str, ln=1)
                
            return pdf.output(dest='S').encode('latin-1')

        if st.button("Generate PDF Report"):
            pdf_bytes = create_pdf()
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="Customer_Segmentation_Report.pdf",
                mime="application/pdf"
            )

# ==========================================
#                  ABOUT
# ==========================================
elif choice == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### AI-Powered Customer Segmentation & Business Intelligence Dashboard
    Built for professional analysis of customer retail behavior. 
    
    **Technologies Used:**
    - Frontend: Streamlit
    - Machine Learning: Scikit-Learn (K-Means Clustering)
    - Visualizations: Plotly & Seaborn
    - Data Processing: Pandas, NumPy
    
    **Features:**
    - End-to-end Machine Learning pipeline.
    - Automated optimal cluster detection.
    - Persona assignment based on clustered behaviors.
    - Real-time customer prediction for marketing strategies.
    """)

