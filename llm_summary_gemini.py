# llm_summary_gemini.py

import google.generativeai as genai
import pandas as pd
import os
import streamlit as st

from dotenv import load_dotenv  # ✅ Add this line

load_dotenv() 
# Set Gemini API Key
genai.configure(api_key=os.getenv("AIzaSyDcMabBEoH7r0yPCGjFDkJMffFYKAcgVAs"))

# Load Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")



# llm_summary_gemini.py  (improved)

import os, pandas as pd, json, textwrap
from dotenv import load_dotenv
import google.generativeai as genai



def _quick_metrics(df: pd.DataFrame) -> dict:
    """Compute hard numbers Pandas is good at."""
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    metrics = {
        "shape": df.shape,
        "missing_by_col": df.isna().sum().to_dict(),
        "num_desc": numeric.describe().round(3).to_dict(),
        "num_skew": numeric.skew().round(3).to_dict(),
        "cat_cardinality": {c: df[c].nunique() for c in categorical.columns},
        "target_candidates": [
            c for c in df.columns if df[c].nunique() < 20 and df[c].dtype != float
        ],
    }

    # find top absolute correlations
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        corr.values[[range(len(corr))]*2] = 0.0      # zero out diagonal
        pairs = (
            corr.stack()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
            .values.tolist()
        )
        metrics["top_corr_pairs"] = pairs
    else:
        metrics["top_corr_pairs"] = []

    return metrics


def get_dataset_summary(df: pd.DataFrame) -> str:
    """Return markdown with stats + Gemini’s commentary."""
    stats = _quick_metrics(df)

    # 1️⃣ Pure stats block (markdown tables)
    md_stats = []

    md_stats.append(f"**Rows × Cols:** `{stats['shape'][0]} × {stats['shape'][1]}`")

    # Missing values
    miss = pd.Series(stats["missing_by_col"]).sort_values(ascending=False)
    md_stats.append("#### Missing values")
    md_stats.append(miss.to_frame("missing").to_markdown())

    # Numeric describe
    num_desc = pd.DataFrame(stats["num_desc"])
    md_stats.append("#### Numeric summary")
    md_stats.append(num_desc.to_markdown())

    # Cardinality
    card = pd.Series(stats["cat_cardinality"]).sort_values(ascending=False)
    if not card.empty:
        md_stats.append("#### Categorical cardinality")
        md_stats.append(card.to_frame("unique").to_markdown())

    # Correlations
    if stats["top_corr_pairs"]:
        md_stats.append("#### Strongest correlations (|ρ|)")
        corr_tbl = pd.DataFrame(
            stats["top_corr_pairs"], columns=["Feature 1", "Feature 2", "|ρ|"]
        )
        md_stats.append(corr_tbl.to_markdown(index=False))

    md_stats_block = "\n\n".join(md_stats)

    # 2️⃣ Ask Gemini to interpret
    prompt = textwrap.dedent(
        f"""
        You are a senior data scientist. Here are computed statistics for a dataset:

        {json.dumps(stats, indent=2)}

        Write an executive summary:
        • highlight the most important red flags or insights (max 8 bullets)  
        • suggest the most suitable ML task (classification / regression / clustering)  
        • mention any preprocessing steps clearly (imputation, scaling, encoding)  
        • keep it concise (<120 words).  
        """
    )

    try:
        chat = model.start_chat()
        commentary = chat.send_message(prompt).text.strip()
    except Exception as e:
        commentary = f"❌ Gemini API Error: {e}"

    # 3️⃣ Return combined markdown
    final_md = (
        "### 📊 **Computed metrics**\n"
        f"{md_stats_block}\n\n---\n\n"
        "### 🧠 **Gemini’s interpretation**\n"
        f"{commentary}"
    )
    return final_md

