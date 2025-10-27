import streamlit as st
from utils import safe_load_image
from config import IMAGES, DEFAULT_IMAGES

def render_education():
    """Render the education section"""
    
    # Load image safely
    image_base64 = safe_load_image(
        IMAGES["education"],
        DEFAULT_IMAGES["education"]
    )
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Orbitron:wght@400..900&family=Poiret+One&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Raleway:ital,wght@0,100..900;1,100..900&display=swap');

        .container-box {{
            background-image: url("{image_base64}");
            background-size: cover;
            background-position: center;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            width: 60%;
            margin: auto;
            text-align: left;
        }}

        .heading8 {{
            font-family: 'Bebas Neue', sans-serif;
            font-size: 30px;
            font-weight: 600;
            text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
            color: rgba(30,144,255);
            letter-spacing: 3px; 
            margin-bottom: 2%;
        }}
        
        .heading2 {{
            font-family: 'orbitron', sans-serif;
            font-size: 15px;
            font-weight: 600;
            text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
            color: rgba(0,0,0);
            text-align: center;
            line-height: 2;
            background: rgba(253,253,253, 0.3);
            border-radius: 5px;
            padding: 4px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            width: 80%;
            margin: auto;
            margin-bottom: 2%;
        }}
        
        .description {{
            font-family: 'Raleway', sans-serif;
            font-weight: 500;
            font-size: 18px;
            color: black;
            line-height: 1.6;
        }}
        </style>

        <div class="container-box">
            <div class="heading8">EDUCATION</div>
            <div class="heading2">
                B TECH (CS) AI AND ML PACIFIC INSTITUE OF TECHNOLOGY UDAIPUR RAJASTHAN [8.5 CGPA]
            </div>
            <div class="heading2">
                10TH FROM PT UMA DUTT PUBLIC SCHOOL CBSE WITH [9.0 CGP]
            </div>
            <div class="heading2">
                12TH FROM PT UMA DUTT PUBLIC SCHOOL CBSE WITH [60%]
            </div>
        </div>
    """, unsafe_allow_html=True)
