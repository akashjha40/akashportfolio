import streamlit as st


# ✅ Required for st.navigation to work
st.set_page_config(page_title="Agent P", page_icon="🕵️",layout="wide")

import pandas as pd


import streamlit as st
import pandas as pd
import plotly.express as px
from llm_summary_gemini import get_dataset_summary



# App header
st.title("🤖 AI-Powered Dataset Analyst")
st.markdown("Upload your dataset")

# File upload
uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])


if uploaded_file:
    try:
        # Read dataset
        
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")

        # Split page into grid: metrics + plots
        col1, col2 = st.columns(2)

        # 🧠 LEFT COLUMN — Gemini-generated report
        with col1:
            st.subheader("📋 Gemini's Analysis")
            with st.spinner("Gemini is analyzing your dataset..."):
                report = get_dataset_summary(df)
                st.markdown(report)

        # 📊 RIGHT COLUMN — Auto Plots
        with col2:
            st.subheader("📈 Visual Insights")

            # Extract numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            option = st.selectbox(
            "Choose the variable to analyze",
            (numeric_cols))

            st.write(f"You selected: {option}")

            for var in numeric_cols:
                if option == var: 
                    # Display basic statistics for the selected variable
                    st.write(f"Basic statistics for {var}:")
                    st.write(df[var].describe())
                    
                else:
                    None

            
            if len(numeric_cols) >= 1:
                # Histogram
                st.plotly_chart(px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}"))

            if len(numeric_cols) >= 2:
                # Correlation Heatmap
                corr = df[numeric_cols].corr().round(2)
                fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig_corr)

                # Scatter Plot of 1st vs 2nd numerical column
                fig_scatter = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                         title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                st.plotly_chart(fig_scatter)

            else:
                st.warning("⚠️ Not enough numerical columns to create full visual insights.")

        # Data preview
        with st.expander("🔍 Preview Uploaded Data"):
            st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")

    
    