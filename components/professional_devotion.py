import streamlit as st
from utils import safe_load_image
from config import IMAGES, DEFAULT_IMAGES

def render_professional_devotion():
    """Render the professional devotion section"""
    
    # Load image safely
    image_base64 = safe_load_image(
        IMAGES["professional_devotion"],
        DEFAULT_IMAGES["professional_devotion"]
    )
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poiret+One&family=Raleway:ital,wght@0,100..900;1,100..900&display=swap');

        .container-box6 {{
            background-image: url("{image_base64}");
            background-size: cover;
            background-position: center;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            width: 60%;
            margin: auto;
            text-align: left;
            margin-bottom: 30px;
        }}

        .heading {{
            font-family: 'Bebas Neue', sans-serif;
            font-size: 30px;
            font-weight: 600;
            text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
            color: rgba(253,253,253,1);
            letter-spacing: 3px; 
            margin-bottom: 2%;
        }}

        .description8 {{
            font-family: 'Raleway', sans-serif;
            font-weight: 400;
            font-size: 18px;
            color: white;
            line-height: 1.6;
            text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            background: rgba(0,0,0,0.2);
            backdrop-filter: blur(2px);
            -webkit-backdrop-filter: blur(2px);
            padding: 12px;
            width: 100%;
            margin: auto;
        }}
        </style>

        <div class="container-box6">
            <div class="heading">Professional Devotion</div>
            <div class="description8">
                I'm a Certified Data Analyst and Data Scientist with a strong background in predictive analytics, machine learning, and AI. Most of what I've learned comes from a lot of hands-on research and experimentation. I've worked on various projects across medical, business, and tech fields, applying data science to solve real-world problems. And i love working and playing with Data.
            </div>
        </div>
    """, unsafe_allow_html=True)
