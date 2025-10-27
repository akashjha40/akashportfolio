import base64
import streamlit as st
from pathlib import Path
from config import DEFAULT_IMAGES

def safe_load_image(image_path, fallback_url=None):
    """
    Safely load and encode image with error handling
    Returns base64 encoded string or fallback URL
    """
    try:
        if isinstance(image_path, (str, Path)):
            image_path = Path(image_path)
        
        if image_path.exists():
            with open(image_path, "rb") as file:
                encoded = base64.b64encode(file.read()).decode()
            return f"data:image/png;base64,{encoded}"
        else:
            st.warning(f"Image not found: {image_path}")
            return fallback_url or DEFAULT_IMAGES.get("background")
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return fallback_url or DEFAULT_IMAGES.get("background")

def safe_load_data(file_path, fallback_data=None):
    """
    Safely load data files with error handling
    """
    try:
        if isinstance(file_path, (str, Path)):
            file_path = Path(file_path)
        
        if file_path.exists():
            import pandas as pd
            return pd.read_csv(file_path)
        else:
            st.warning(f"Data file not found: {file_path}")
            return fallback_data
    except Exception as e:
        st.error(f"Error loading data file {file_path}: {e}")
        return fallback_data

def safe_load_model(model_path):
    """
    Safely load machine learning models with error handling
    """
    try:
        if isinstance(model_path, (str, Path)):
            model_path = Path(model_path)
        
        if model_path.exists():
            import joblib
            return joblib.load(model_path)
        else:
            st.warning(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        return None

def create_glass_box_style():
    """Create reusable glass box styling"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Audiowide&family=Bebas+Neue&family=Kosugi+Maru&family=Noto+Sans:ital,wght@0,100..900;1,100..900&family=Poiret+One&display=swap');

    .glass-box {
        font-family: 'Audiowide', sans-serif;
        font-size: 50px;
        font-weight: 550;  
        text-align: center;
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.4);
        color: rgba(35,43,43);
    }
    </style>
    """

def create_blue_box_style():
    """Create reusable blue box styling"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Kosugi+Maru&family=Noto+Sans:ital,wght@0,100..900;1,100..900&family=Poiret+One&display=swap');
    .blue-box{ 
        font-family: 'Poiret One', sans-serif;
        font-size: 20px;
        font-weight: 700;  
        text-align: center;
        opacity: 0.8;
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
        color: WHITE;
        background: rgba(0,112,255, 0.8);
        padding: 10px;
        border-radius: 5px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        width: 25%;
        margin: auto;
    }
    </style>
    """
