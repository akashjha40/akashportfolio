import streamlit as st
from utils import safe_load_image
from config import IMAGES, DEFAULT_IMAGES

def render_certifications():
    """Render the certifications section"""
    
    # Load image safely
    image_base64 = safe_load_image(
        IMAGES["certifications"],
        DEFAULT_IMAGES["certifications"]
    )
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Orbitron:wght@400..900&family=Poiret+One&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Raleway:ital,wght@0,100..900;1,100..900&display=swap');

        .container-box2 {{
            background-image: url("{image_base64}");
            background-size: cover;
            background-position: center;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            align: center;
            width: 60%;
            margin: auto;
            margin-top: 30px;
            text-align: left;
            margin-bottom: 10px;
        }}

        .heading3 {{
            font-family: 'Bebas Neue', sans-serif;
            font-size: 30px;
            font-weight: 600;
            text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.1);
            color: rgba(255,160,122);
            text-align: left;
            letter-spacing: 2px;
            margin-bottom: 2%;
        }}

        .description4 {{
            font-family: 'orbitron', sans-serif;
            font-weight: 600;
            font-size: 21px;
            color: white;
            text-align: center;
            line-height: 1.6;
            text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.4);
            padding: 10px;
            border-radius: 5px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}
        </style>

        <div class="container-box2">
            <div class="heading3">CERTIFICATIONS</div>
            <div class="description4">
                AI & ML AS WELL AS DATA SCIENCE & ANALYTICS
            </div>
        </div>
    """, unsafe_allow_html=True)
