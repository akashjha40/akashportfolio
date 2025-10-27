import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

# Page configuration
st.set_page_config(page_title="Portfolio", layout='wide', page_icon="📂")

# Import our custom components and utilities
from components.header import render_header
from components.professional_devotion import render_professional_devotion
from components.education import render_education
from components.certifications import render_certifications
from components.skills import render_skills
from components.projects import render_projects
from utils import safe_load_image, safe_load_data, safe_load_model
from config import IMAGES, DATA_FILES, MODELS, CERTIFICATES, DEFAULT_IMAGES

# Load external CSS safely
try:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning("CSS file not found, using default styling")
except Exception as e:
    st.warning(f"Could not load CSS: {e}")

# Add custom CSS for expanders
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Orbitron:wght@400..900&family=Poiret+One&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Raleway:ital,wght@0,100..900;1,100..900&display=swap');
    
    /* Style for the expander */
    details {
        background-color: rgba(253,253,253, 0.5) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
        width: 100% !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        padding: 10px !important;
    }
    
    div[data-testid="stExpander"] summary {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        color: rgba(0, 0, 0, 1) !important;
        text-align: left !important;
    }
    
    /* Style for the expander content */
    div[data-testid="stExpanderContent"] p {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: rgba(0,0,0,1) !important;
        text-align: left !important;
    }
    
    /* Style for list items inside expander */
    div[data-testid="stExpanderContent"] ul li,
    div[data-testid="stExpanderContent"] ol li,
    div[data-testid="stExpanderContent"] li p {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: rgba(0,0,0,1) !important;
        text-align: left !important;
    }
    
    /* Additional selector for any text elements inside the expander */
    div[data-testid="stExpanderContent"] div {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: rgba(0,0,0,1) !important;
        text-align: left !important;
    }
    </style>
""", unsafe_allow_html=True)

# Render the main header section
render_header()

# Render the professional devotion section
render_professional_devotion()

# Render the education section
render_education()

# Render the certifications section
render_certifications()

# Create columns layout for certificates expander
col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is wider

# Create Certificates Expander
with col2:
    expander = st.expander("View Certificates")
    
    # Load Images (you may need to update these paths)
    try:
        # Try multiple possible paths for certificates
        cert_paths = [
            r"D:\Akash\CERTIFICATIONS\00537fe6-98d8-4f59-bc69-bf88f5d1c4ab.png",
            r"C:\Users\akash\CERTIFICATIONS\00537fe6-98d8-4f59-bc69-bf88f5d1c4ab.png",
            "assets/cert1.png",  # Fallback to assets folder
            "assets/cert2.png"
        ]
        
        cert_loaded = False
        for cert_path in cert_paths:
            try:
                expander.image(cert_path, width=500)
                cert_loaded = True
                break
            except:
                continue
                
        if not cert_loaded:
            expander.info("Certificate images would be displayed here")
            expander.write("Please update the certificate image paths in the code")
            
    except Exception as e:
        expander.warning(f"Could not load certificate images: {e}")
        expander.info("Certificate images would be displayed here")

st.write("")
st.write("")

# Render the skills section
render_skills()

# Render the projects section
render_projects()

# Add some spacing
st.write("")
st.write("")

# Helper function for plotting
def plot_actual_vs_predicted(df, actual_col='Original', predicted_col='Predicted', title="Actual vs Predicted"):
    fig = px.scatter(df, x=actual_col, y=predicted_col, title=title,
                     labels={actual_col: "Actual", predicted_col: "Predicted"})
    
    # Add 45-degree reference line (y = x)
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())
    
    fig.add_trace(
        px.line(x=[min_val, max_val], y=[min_val, max_val]).data[0]
    )
    
    fig.update_traces(line=dict(color='red', dash='dash'), selector=dict(type='scatter'))
    
    st.plotly_chart(fig, use_container_width=True)

# HOUSE PRICES PROJECT EXPANDER
with st.expander("ADVANCED REGRESSION MODEL FOR HOUSE PRICES DETAILS"):
    st.write("""
HEY SO IN THIS PROJECT I USED LINEAR REGRESSION AND RANDOM FOREST

•  The goal was to make predictive model to estimate housing prices using a comprehensive dataset.

• **Techniques & Methods:**
  - Conducted extensive data cleaning and exploratory data analysis to understand key patterns.
  - Performed feature engineering to enhance model performance.
  - Applied manual feature selection using RFE ,VIF and p-value methods to reduce multicollinearity and ensure statistical significance.
  - Transformed categorical variables using Dummy encoding.
  - Experimented with multiple machine learning algorithms including linear regression, decision trees, and ensemble methods (e.g., Random Forest and Gradient Boosting).
  - Applied cross-validation and hyperparameter tuning to optimize the model.

• **Results:**
    - Out of the 80 columns in the dataset, only 7 were found to significantly explain the majority of the variance. 
    - Achieved a high level of predictive accuracy with an R² score of approximately 87%, demonstrating the effectiveness of the modeling approach.

• I REALLY ENJOYED WORKING ON THIS AND I GOT TO LEARN A LOT ABOUT FEATURE SELECTION METHODS
""")
    
    try:
        # Load house prices data
        df = pd.read_csv(r"D:\Akash\project\Predicted1.csv")
        residuals = df['Original'] - df['Predicted']
        
        klm1, klm2, klm3 = st.columns(3)
        
        with klm1:
            # Create scatter plot
            plot_actual_vs_predicted(df, actual_col='Original', predicted_col='Predicted')
            
        with klm3:
            # Residuals vs Predicted plot
            fig = px.scatter(df, x='Predicted', y=residuals, title='Residuals vs Predicted',
                         labels={'Predicted': 'Predicted', 'y': 'Residuals'})
            fig.add_hline(y=0, line_dash="dash", line_color='red')
            st.plotly_chart(fig)
        
        with klm2:
            # Residuals distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=residuals, histnorm='probability density', name='Residuals'))
            kde_x = np.linspace(min(residuals), max(residuals), 100)
            kde_y = stats.gaussian_kde(residuals)(kde_x)
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name='KDE', line=dict(color='blue')))
            fig.update_layout(title='Residuals Distribution', xaxis_title='Residual', yaxis_title='Density')
            st.plotly_chart(fig)
            
        with klm1:
            st.metric("R2 SCORE", 0.87)
            
        with klm1:
            mse = (residuals ** 2).mean()
            rmse = round(mse ** 0.5)
            st.metric("RMSE", rmse)
            
    except Exception as e:
        st.warning(f"Could not load house prices data: {e}")
        st.info("House prices analysis plots would be displayed here")

# HEART DISEASE PROJECT EXPANDER
with st.expander("CLASSIFICATION MODEL FOR PREDICTING CHRONIC HEART DISEASE"):
    st.write("""
• **Project Overview:**
  - Developed a classification model to predict chronic heart disease using the Framingham Heart Study dataset.
  - Applied logistic regression and other classification algorithms to identify risk factors.

• **Key Features Used:**
  - Age, education, smoking status, blood pressure medications
  - Total cholesterol, systolic/diastolic blood pressure
  - BMI, heart rate, glucose levels
  - 10-year risk of coronary heart disease (CHD)

• **Methodology:**
  - Data preprocessing and handling missing values
  - Feature scaling using StandardScaler
  - Train-test split (70-30)
  - Model evaluation and performance metrics

• **Results:**
  - Successfully identified key risk factors for heart disease
  - Model performance metrics available in the analysis
""")
    
    try:
        # Load heart disease data
        df = pd.read_csv(r"C:\Users\akash\Downloads\archive\framingham.csv")
        df2 = df.dropna()
        
        # Display basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"Total samples: {len(df)}")
            st.write(f"Samples after cleaning: {len(df2)}")
            st.write(f"Features: {len(df.columns)}")
            
        with col2:
            st.subheader("Key Features")
            st.write("• Age, Education, Smoking Status")
            st.write("• Blood Pressure, Cholesterol")
            st.write("• BMI, Heart Rate, Glucose")
            st.write("• 10-Year CHD Risk")
            
    except Exception as e:
        st.warning(f"Could not load heart disease data: {e}")
        st.info("Heart disease analysis would be displayed here")

# NGO CLUSTERING PROJECT EXPANDER
with st.expander("K-MEANS CLUSTERING ON NGO DATA"):
    st.write("""
• **Project Objective:**
  - Performed K-means clustering on NGO data to identify countries in greatest need of aid
  - Enabled more effective resource allocation for humanitarian assistance

• **Methodology:**
  - Applied K-means clustering algorithm
  - Identified optimal number of clusters using elbow method
  - Analyzed cluster characteristics and needs assessment

• **Key Insights:**
  - Clustered countries based on development indicators
  - Prioritized aid distribution based on cluster analysis
  - Improved resource allocation efficiency

• **Impact:**
  - Better targeting of humanitarian aid
  - Data-driven approach to international development
  - Enhanced NGO operational efficiency
""")

# GENE EXPRESSION PROJECT EXPANDER
with st.expander("CONTROL ANALYSIS ON GSM GENE EXPRESSION DATASET"):
    st.write("""
• **Research Focus:**
  - Conducted case-control analysis on GSM gene expression dataset of pancreatic cancer
  - Identified differential gene expression patterns between cases and controls

• **Methodology:**
  - Applied statistical analysis to gene expression data
  - Identified significantly differentially expressed genes
  - Performed pathway analysis and functional annotation

• **Key Findings:**
  - Discovered novel gene expression patterns
  - Identified potential biomarkers for pancreatic cancer
  - Enhanced understanding of cancer biology

• **Applications:**
  - Potential diagnostic markers
  - Therapeutic target identification
  - Cancer research advancement
""")

# MINOR PROJECTS EXPANDER
with st.expander("MINOR PROJECTS"):
    st.write("""
• **Data Preprocessing Projects:**
  - Implemented comprehensive data cleaning pipelines
  - Applied various preprocessing techniques for different datasets
  - Developed standardized data preparation workflows

• **Principal Component Analysis (PCA):**
  - Applied PCA for dimensionality reduction
  - Analyzed variance explained by principal components
  - Visualized data in reduced dimensional space

• **Techniques Applied:**
  - Missing value handling and imputation
  - Outlier detection and treatment
  - Feature scaling and normalization
  - Data quality assessment and validation

• **Tools and Technologies:**
  - Python, Pandas, NumPy
  - Scikit-learn for machine learning
  - Matplotlib and Seaborn for visualization
  - Statistical analysis and hypothesis testing
""")


