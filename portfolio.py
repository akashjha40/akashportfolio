import base64
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Page configuration
st.set_page_config(page_title="Portfolio", layout='wide', page_icon="📂")

# Import our custom components and utilities
from components.header import render_header
from components.professional_devotion import render_professional_devotion
from components.education import render_education
from components.certifications import render_certifications
from components.skills import render_skills
from utils import safe_load_image, safe_load_data, safe_load_model
from config import IMAGES, DATA_FILES, MODELS, CERTIFICATES, DEFAULT_IMAGES




import pathlib
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

# Render the main header section
render_header()

# Render the professional devotion section
render_professional_devotion()


# Render the education section
render_education()

# Render the certifications section
render_certifications()

# Create columns layout (3 columns)
col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is wider

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

import st_tailwind as tw
tw.initialize_tailwind()
# Create Expander
with col2:
    expander = st.expander("View Certificates")
    
# Load Images
    expander.image(r"D:\Akash\CERTIFICATIONS\00537fe6-98d8-4f59-bc69-bf88f5d1c4ab.png", width=500)
    expander.image(r"D:\Akash\CERTIFICATIONS\07fe408d-1966-4ed7-b5cd-0509aa7a2067.png", width=500)

st.write("")
st.write("")

# Render the skills section
render_skills()
@st.cache_data(ttl=3600)
def get_base64_image(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    return f"data:image/png;base64,{encoded}"



#expander inside project-----------------------------
    



image_path = r"C:\Users\akash\Downloads\compressor\top-view-desk-with-copy-space.jpg"
image_base64 = get_base64_image(image_path)
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Orbitron:wght@400..900&family=Poiret+One&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Raleway:ital,wght@0,100..900;1,100..900&family=Roboto+Condensed:ital,wght@0,100..900;1,100..900&display=swap');

    .box-contain {{
        background-image: url("{image_base64}");
        background-size: cover;
        background-position: center;
        padding: 50px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        width: 100%;
        padding: 10px;
        margin: auto;
        margin-top: 5%;
        margin-bottom: 1%;
        text-align: center;    
    }}
    .expand {{
        background: rgba(253,253,253,0.5);
        background-size: cover;
        background-position: center;
        padding: 50px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        width: 70%;
        margin: auto;
        margin-top: 5%;
        text-align: center;
        }} 
    .heading5 {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 30px;
        font-weight: 600;
        letter-spacing: 3px; 
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.1);
        color: rgba(0,0,0,0.8);
        text-align: left;
        text-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
        padding:20px;
        width: 100%;
        
        
        
    }}

    .description5 {{
        font-family: 'Roboto Condensed', sans-serif;
        font-weight: 600;
        font-size: 18px;
        color: rgba(18,53,36);
        # text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3);
        text-align: left;
        line-height: 1.6;
        
        
        width: 100%;
        margin: auto;
        PADDING:20PX;
    }}
    </style>

    <div class="box-contain">
        <div class="heading5">PROJECTS AND EXPERIENCE</div>
        <div class="description5">• ADVANCED REGRESSION MODEL FOR PREDICTING HOUSE PRICES</div>
        <div class="description5">• CLASSIFICATION MODEL FOR PREDICTING CHRONIC HEART DISEASE</div>
        <div class="description5">• PERFORMED K-MEANS CLUSTERING ON NGO DATA TO IDENTIFY COUNTRIES IN GREATEST NEED OF AID, ENABLING MORE EFFECTIVE RESOURCE ALLOCATION</div>
        <div class="description5">• CONDUCTED CASE -  CONTROL ANALYSIS ON GSM GENE EXPRESSION DATASET OF PANCREATIC CANCER TO IDENTIFY DIFFERENTIAL GENE EXPRESSION PATTERNS</div>    
         <div class="description5">• MINOR PROJECTS -  DATA PREPROCESSING, CLEANING AND PCA ON VARIOUS DATASETS</div>
    </div>
""", unsafe_allow_html=True)

# Ensure Expander is inside the same div
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df= pd.read_csv(r"C:\Users\akash\Downloads\archive\framingham.csv")



df2 = df.dropna()


cols= ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']



# import matplotlib.pyplot as plt

# # Group by 'currentSmoker' and calculate the mean of the 'glucose' column
# smoke_glucose_mean = df2.groupby('currentSmoker')['glucose'].mean()

# # Plot the bar graph
# smoke_glucose_mean.plot(kind='bar', color=['blue', 'orange'])

# # Set labels and title
# plt.xlabel('Current Smoker')
# plt.ylabel('Average Glucose')
# plt.title('Average Glucose Levels by Smoking Status')

# # Show the plot
# plt.show()


# # In[29]:


# # # Group by 'currentSmoker' and calculate the mean of the 'glucose' column
# # smoke_glucose_mean = df2.groupby('TenYearCHD')['totChol'].mean()

# # # Plot the bar graph
# # smoke_glucose_mean.plot(kind='bar', color=['green', 'pink'])

# # # Set labels and title
# # plt.xlabel('CHD')
# # plt.ylabel('total chol')
# # plt.title('Average chol')

# # # Show the plot
# # plt.show()




df3 = df2.copy() 


# # In[35]:




# # In[36]:





# # In[37]:





# # In[38]:


# # 'age', 'education','cigsPerDay', 'totChol', 'sysBP',
# #        'diaBP', 'BMI', 'heartRate', 'glucose'




import sklearn
from sklearn.model_selection import train_test_split
import statsmodels
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics




scale = StandardScaler()


# In[41]:


df3[['age', 'education','cigsPerDay', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']] = scale.fit_transform(df3[['age', 'education','cigsPerDay', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']])





train,test = train_test_split(df3,train_size=0.7,test_size=0.3, random_state=100)


# In[54]:


lmx = train.drop(columns=['TenYearCHD'],axis=1)


# In[55]:


lmy = train['TenYearCHD']
ftx= train.drop(columns=['TenYearCHD'])
fty = train['TenYearCHD']

# In[58]:


# plt.figure(figsize=(16,10))
# sns.heatmap(lmx.corr(),annot=True, cmap="YlGnBu")
# plt.title("After removing highly correlated variables")

# # In[59]:


# ltx = test.drop(columns=['TenYearCHD','cigsPerDay', 'prevalentHyp', 'sysBP', 'BMI', 'heartRate', 'glucose'],axis=1)
lty = test['TenYearCHD']


# In[60]:


rdt_X= test[['male', 'age', 'education', 'currentSmoker', 'BPMeds',
       'prevalentStroke', 'diabetes', 'totChol', 'diaBP']] 



from sklearn.linear_model import LogisticRegression


# In[67]:


from sklearn.feature_selection import RFE


# In[68]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from IPython.display import SVG

from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[69]:

df3_X = df3[['male', 'age', 'education', 'currentSmoker', 'BPMeds',
       'prevalentStroke', 'diabetes', 'totChol', 'diaBP']]
df3_y = df3[['TenYearCHD']]


# In[164]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(df3_X, df3_y)


# In[ ]:





# In[165]:


y_sm=pd.DataFrame(y_sm)


# In[166]:


X_train, X_test, y_train, y_test = train_test_split( X_sm,y_sm , test_size = 0.2, random_state = 0) 


# In[175]:
import joblib

import joblib

# Load the model from disk
def classifier(classifier_path):
    return joblib.load(classifier_path)

# Make predictions using the loaded model

def predictions(model, dataset):
    return model.predict(dataset)

# ✅ Use it correctly
model = classifier(r"C:\Users\akash\logist.pkl")

y_train_preds_rf = predictions(model, X_train)
y_test_preds_rf  = predictions(model, X_test)



# #### PERFORMANCE TEST IN NON SMOTE ACTUAL NEW DATA

# In[178]:


x22 = ftx[['male', 'age', 'education', 'currentSmoker', 'BPMeds',
       'prevalentStroke', 'diabetes', 'totChol', 'diaBP']]


# In[179]:


y22 = model.predict(x22)

ypredict_pro=model.predict_proba(x22)[:,1]
acc2 = accuracy_score(fty,y22)
fpr2, tpr2, thresholds2 = roc_curve(fty, y22)

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
roc_auc = roc_auc_score(fty,ypredict_pro)
x22['pred2'] = y22
x22['original2']= fty
X_train['Pred'] = y_train_preds_rf
X_train['Original'] = y_train

import plotly.express as px

with st.expander("CLASSIFICATION MODEL FOR PREDICTING CHRONIC HEART DISEASE"):
    tw.write("""
- **Data Cleaning & Preparation:**
    - Reviewed all variables and addressed missing values by either dropping rows or imputing medians based on the percentage of missing data.
    - Standardized the dataset using Standard Scaler.
- **Exploratory Data Analysis (EDA):**
    - Investigated relationships between features like heart rate, cholesterol, and CHD.
    - Determined the average age of CHD patients.
    - Created a correlation heatmap to identify and drop highly correlated features.
- **Initial Modeling:**
    - Built a GLM model.
    - Removed features with p-values > 0.05 and VIF > 5, achieving an 85% accuracy.
    - Noticed a high false negative rate, which is critical for a disease prediction model.
- **Model Optimization:**
    - Applied SMOTE to balance the class distribution.
    - Utilized a probability cutoff plot to optimize the classification threshold.
    - Switched to a Random Forest Classifier and fine-tuned parameters via grid search.
    - Reduced the false negative rate to 3%, with a ROC score of 95%, 95% training accuracy, and 90% test accuracy.
""",classes="text-lg text-black font-semibold")
    
    colc1, colc2, colc3 = st.columns(3)
    
    st.subheader("📊 Model Performance Metrics On Test Data")
    st.metric(label="Accuracy", value=f"{acc2:.2%}")
    st.metric(label="ROC AUC Score", value=f"{roc_auc * 100 :.2f}%")
    with colc1:
        st.dataframe(X_train.head(10))
        
        
    with colc2:
        st.image(r"C:\Users\akash\Downloads\compressor\Screenshot 2025-03-21 065338.png")    
        

    with colc3:
# In[401]:

        cm = metrics.confusion_matrix(y_train, y_train_preds_rf)
        TP2 = cm[1,1] # true positive 
        TN2 = cm[0,0] # true negatives
        FP2 = cm[0,1] # false positives
        FN2 = cm[1,0] # false negatives
        FNR2 = (FN2/float(FN2+TP2)*100)
        
        st.subheader("🌀 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)

    st.subheader(f"⚠ **False Negative Rate (FNR): 2.9%")
    

with st.expander("K-MEANS CLUSTERING ON NGO DATA"):
    tw.write("""
- **Data Processing & Cleaning:**
  - Loaded the raw dataset and performed initial cleaning to ensure high data quality.

- **Feature Engineering:**
  - Created new metrics such as the percentage of GDP spent on health and the percentage of GDP on imports/exports to normalize data relative to each country's economy.

- **Exploratory Data Analysis (EDA):**
  - Analyzed the dataset and discovered that North America shows high health spending, primarily due to expensive healthcare costs, with the US as a notable example.

- **Feature Selection & Multicollinearity Check:**
  - Addressed multicollinearity among features and ensured that only the most relevant variables were selected.
  - Verified the dataset’s clusterability using the Hopkins statistic.

- **Determining Optimal Clusters:**
  - Employed the elbow method to identify the optimal number of clusters for meaningful segmentation.

- **Outlier Handling:**
  - Removed outliers during model creation to avoid skewed results, but retained the outlier countries for later analysis.

- **K-Means Clustering & Final Integration:**
  - Applied the K-Means algorithm to cluster countries based on their health and financial conditions.
  - Merged the clustering results back with the original dataset, including the outlier countries, to provide a complete picture of which countries need the most aid.
""",classes="text-lg text-black font-semibold")
    def load_data():
        return pd.read_csv(r"C:\Users\akash\Downloads\compressor\clust_slit.csv")

    ngo = load_data()

# Select relevant columns for clustering
    col = ngo.columns[1:]

    st.title("Data")
    st.dataframe(ngo)

# Streamlit UI elements
    st.title("Cluster Analytics - Box Plots")
# Initialize session state
    cols1, cols2, cols3 = st.columns(3)
# Function to create and display box plots
    
    with cols1:

        def plot_boxplot(y_variable):
            fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
            sns.boxplot(x='ClusterID', y=y_variable, data=ngo, ax=ax, width=0.4,palette='pastel',color='skyblue')
            # ax.set_title(f'Box Plot of {y_variable}')
            
            st.pyplot(fig)
            plt.show()

    # Buttons to trigger specific plots
        
        
    
    with cols1:
        option = st.selectbox('select variable', (col))
        if option == 'exports':
            plot_boxplot(ngo['exports'])
        elif option == 'imports':
            plot_boxplot(ngo['imports'])
        elif option == 'gdpp':
            plot_boxplot(ngo['gdpp'])
        elif option == 'income':
            plot_boxplot(ngo['income'])
        elif option == 'health':
            plot_boxplot(ngo['health'])
        elif option == 'inflation':
            plot_boxplot(ngo['inflation'])
        elif option == 'total_fir':
            plot_boxplot(ngo['total_fer'])
        elif option == 'life_expec':
            plot_boxplot(ngo['life_expec'])
        elif option == 'child_mort':
            plot_boxplot(ngo['child_mort'])
        else:
            plot_boxplot(ngo['child_mort'])
    
    ngo2 = pd.read_csv(r"D:\Akash\clust. proj\Country-data.csv")
    # ngo22 = st.dataframe(ngo2)
    with cols2:
        fig = px.choropleth(
        ngo2,
        locations="country",
        locationmode="country names",
        color="health",
        color_continuous_scale="plasma",
        title="Total health spending per capita. Given as percentage of GDP per capita",
        template="plotly_dark"
    )
        fig.update_layout(height=700, width=1200)
    # Streamlit App
        st.title("🌍 World Health spending Visualization")
        st.plotly_chart(fig, use_container_width=True)
    
with st.expander("Children Mental Health Analysis on NSCH Data 2022-2023"):
    tw.write("""
### Factors Affecting Children's Mental Health

- I conducted a statistical analysis to explore the factors affecting children's mental health, aiming for early detection and better care.  
- This topic is personal to me because I have seen how mental health challenges can deeply impact children's lifestyle and health.  

**Data Source:** National Survey of Children's Health (NSCH) 📊  

- I analyzed key indicators from the NSCH dataset to understand how bullying, friendship difficulties, and sleep patterns relate to childhood mental health.  
- These insights are vital for recognizing early signs and taking proactive steps to raise emotionally healthy children.  

---

### ⚙️ Data Preparation
- The original dataset contained **885 variables (columns)**, which required careful selection of relevant indicators.  
- I referred to the official **data dictionary** to identify and choose the correct variables aligned with mental health outcomes.  
- Many variables were coded with numeric values (e.g., `1 = No`, `2 = Maybe`, `3 = Yes`).  
- To ensure clarity and statistical validity, I **recoded these into binary formats** such as **Yes/No** for easier interpretation.  
- This preprocessing step was crucial to make the analysis more reliable and the findings more actionable.  

---

🔹 **Plot 1: Bullying and Depression**  
- 20.5% of bullied children suffer from depression, compared to 3.6% of non-bullied children.  
- **Conclusion:** Bullying increases the risk of depression over 5 times.  

🔹 **Plot 2: Bullying and Anxiety**  
- Children who are bullied are 4 times more likely to experience anxiety than those who are not.  
- **Conclusion:** Bullying is a major contributor to childhood anxiety and emotional distress.  

🔹 **Plot 3: Difficulty Making Friends and Depression**  
- Children with difficulty making friends are 8.46 times more likely to be depressed.  
- 19.39% of such children have depression, versus just 2.77% without such difficulty.  
- **Conclusion:** Social isolation is a critical risk factor for childhood mental illness.  

🔹 **Plot 4: Sleep Patterns and Depression**  
- 43.5% of depressed children experience poor sleep, compared to 31.4% of non-depressed children.  
- Children with poor sleep are 1.68 times more likely to experience depression.  
- **Conclusion:** Sleep quality is a strong behavioral indicator of mental well-being.  

---

### 🔍 Why These Metrics Matter
- These insights do not just show correlations — they provide actionable direction.  
- They tell us:  

  - ✅ **What to do:** Encourage open communication, build safe school environments, and ensure children have supportive friendships and good sleep routines.  
  - 🚩 **What not to ignore:** Warning signs like bad grades, refusal to go to school, social withdrawal, or trouble sleeping may indicate underlying issues like bullying, anxiety, or depression.  

---

### 📌 Tools & Methods
- Used **Python and Plotly** for visualization.  
- Applied statistical measures such as **Odds Ratio**, **Relative Risk**, and **Probability** to calculate the findings.  
""",classes="text-lg text-black font-semibold")
    nsch = pd.read_csv(r"D:\Akash\VS\Portfolio\assets\Nsch.csv")
    
    def plot_stacked_sleep_bar(df, width=700, height=600, colors=None, title='', x_label='', y_label='', legend=None):
        fig_n = go.Figure()

        for col, color in zip(df.columns, colors):
            fig_n.add_bar(
                x=df.index,
                y=df[col],
                name=col,
                marker_color=color
            )
        
        fig_n.update_layout(
            barmode='stack',
            width=width,
            height=height,
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend=legend
        )

        return fig_n
        
    def calc(df: pd.DataFrame) -> pd.DataFrame:
 
        percent = df.div(df.sum(axis=1), axis=0) * 100
        return percent
    
    
    bullying = nsch.groupby(['bullied_2223', 'Depression']).size().unstack()

    perc = calc(bullying)
    clm1, clm2, clm3 = st.columns(3)
    with clm1:
        fig = plot_stacked_sleep_bar(
        perc,
        width=700,
        height=600,
        colors=['#7B68EE', '#FF5733'],
        title='Bullied and Depressed',
        x_label='Depressed',
        y_label='Bullied',
        legend=dict(title='Depressed')
    )
        st.plotly_chart(fig, use_container_width=True, key="bullying_depression_chart")

        









with st.expander("CONTROL ANALYSIS ON GSM GENE EXPRESSION DATASET"):
    tw.write("""
- **Imported essential libraries** for data analysis and visualization, including tools for statistical modeling and plotting.  
- **Loaded and preprocessed the gene expression dataset**, ensuring proper formatting, handling missing values, and preparing the data for analysis.  
- **Annotated sample groups** by classifying them into 'case' and 'control' categories to enable comparative analysis.  
- **Explored the raw data** using visual tools like boxplots, heatmaps, and PCA to identify outliers, check distribution patterns, and assess overall data quality.  
- **Normalized the expression values** to reduce technical variation and make data from different samples comparable.  
- **Performed differential expression analysis** using statistical tests to identify genes that are significantly upregulated or downregulated between case and control groups.  
- **Applied multiple testing correction (FDR)** to account for the large number of comparisons and reduce the chance of false positives.  
- **Filtered significant genes** based on adjusted p-values and fold change thresholds to focus on the most biologically relevant results.  
- **Visualized key findings** using volcano plots and clustered heatmaps to highlight differentially expressed genes and overall expression patterns.  
- **Exported results** for downstream use, including further pathway analysis or integration into other pipelines.    
""",classes="text-lg text-black font-semibold")
    colz1, colz2, colz3 = st.columns(3)

    with colz1:
        st.image(r"C:\Users\akash\Downloads\compressor\Screenshot 2025-03-21 213836.png")
    with colz2:
        st.image(r"C:\Users\akash\Downloads\compressor\Screenshot 2025-03-21 213806.png")
    with colz3:
        gsm = pd.read_csv(r"C:\Users\akash\Downloads\compressor\gsm_Final_2.csv")
        st.dataframe(gsm)

with st.expander("MINOR PROJECTS"):
    tw.write("""
- **Performed data cleaning, preprocessing, and feature engineering** on various real-world datasets including IMDb movie data, Google Play Store apps dataset, Housing price data, and Telecom churn records.  
- **Handled missing values, outliers, and categorical encoding** to prepare datasets for effective modeling and analysis.  
- **Applied data aggregation and grouping techniques** to extract meaningful insights, such as customer churn behavior, app popularity trends, and housing market patterns.  
- **Implemented dimensionality reduction techniques** such as **Principal Component Analysis (PCA)** on gene expression data and other high-dimensional datasets to improve model performance and visualization.  
- **Explored multivariate relationships and trends** using statistical summaries, correlation analysis, and visualizations.  
- Projects served as foundational practice for **EDA, data wrangling, and unsupervised learning workflows**, supporting further machine learning tasks.
""",classes="text-lg text-black font-semibold")

# st.markdown("""

#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Orbitron:wght@400..900&family=Poiret+One&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Raleway:ital,wght@0,100..900;1,100..900&family=Roboto+Condensed:ital,wght@0,100..900;1,100..900&display=swap');

#         .box-tech {
#                 background: rgba(253,253,253,0.5);
#                 padding: 10px;
#                 border-radius: 15px;
#                 width: 20%;
#                 margin: auto;
#                 align: center;

#               }
#         .techs {
#               font-family: 'Roboto Condensed', sans-serif; 
#               font-size: 18px;
#               font-weight: 500;
#               color: rgba(0,0,0,1);
#               text-align: center;
#               text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.1);
#               }
#         </style>
#             <div class="box-tech">
#         <div class="techs">SKILLS</div>
#     </div> 
#         """,unsafe_allow_html=True)
import streamlit as st

import streamlit as st

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Raleway:wght@100..900&display=swap');

    .skills-boxx {
        background: rgba(253,253,253,0.5);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        width: 80%;
        margin: auto;
        text-align: center;
    }

    .skills-headingss {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 40px;
        font-weight: 600;
        color: rgba(253,253,253,1);
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
        letter-spacing: 4px;
        margin-bottom: 30px;
    }

    .about-items {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 20px;
        font-weight: 300;
        color: rgba(74, 74, 74);
        text-align: center;
        letter-spacing: 1px;
        background: rgba(253,253,253,0.8);
        border-radius: 10px;
        padding: 10px;
        width: 80%;
        margin: 10px auto;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .about-items span {
        font-weight: 700;
        color: #0078ff;
    }

    .about-links {
        margin-top: 20px;
    }

    .about-link-btn {
        text-decoration: none;
        color: white;
        background: #0078ff;
        padding: 10px 18px;
        border-radius: 8px;
        margin: 8px;
        display: inline-block;
        font-family: 'Raleway', sans-serif;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    .about-link-btn:hover {
        background: #005fcc;
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.15);
    }
    </style>

    <div class="skills-boxx">
        <div class="skills-headingss">ABOUT ME</div>
        <div class="about-items"><span>Name:</span> Akash Jha</div>
        <div class="about-items"><span>City:</span> Udaipur</div>
        <div class="about-items"><span>College:</span> Pacific Institute of Technology, Udaipur</div>
        <div class="about-items"><span>Year:</span> 4th</div>
        <div class="about-items"><span>Course:</span> B.Tech in Computer Science, Data Science & AI</div>

    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st

# Custom CSS for box, font, and layout
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');

    .about-container {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 20px;
        padding: 30px;
        margin-top: 40px;
        width: 85%;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .about-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 45px;
        letter-spacing: 3px;
        color: #0078ff;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }

    .about-info {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 20px;
        color: #333;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }

    /* Style for Streamlit link buttons */
    div[data-testid="stLinkButton"] button {
        background-color: #0078ff !important;
        color: white !important;
        border-radius: 10px !important;
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 18px !important;
        padding: 10px 25px !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="stLinkButton"] button:hover {
        background-color: #005fcc !important;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# Main About Box
with st.container():

    col1, col2 = st.columns(2)
    with col1:
        st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/akashjha40")
    with col2:
        st.link_button("💻 GitHub", "https://github.com/akashjha40/MyDataProjects")

    




