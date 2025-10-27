import streamlit as st
from utils import create_glass_box_style, create_blue_box_style, safe_load_image
from config import IMAGES, DEFAULT_IMAGES

def render_header():
    """Render the main header section with title and subtitle"""
    
    # Load background image safely
    background_image = safe_load_image(
        IMAGES["background"], 
        DEFAULT_IMAGES["background"]
    )
    
    # Set background
    if background_image.startswith("data:"):
        # Local image - set as background
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{background_image}");
            background-size: cover;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        # Fallback URL - use as background
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{background_image}");
            background-size: cover;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    # Main title
    st.markdown(create_glass_box_style(), unsafe_allow_html=True)
    st.markdown("""
        <div class="glass-box">AKASH JHA PORTFOLIO</div>
    """, unsafe_allow_html=True)
    
    # Subtitle
    st.markdown(create_blue_box_style(), unsafe_allow_html=True)
    st.markdown("""
        <div class="blue-box">DATA SCIENCE AND ANALYTICS</div>
    """, unsafe_allow_html=True)
    
    # Add spacing
    for _ in range(4):
        st.write("")
