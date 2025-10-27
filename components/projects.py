import streamlit as st
from utils import safe_load_image
from config import IMAGES, DEFAULT_IMAGES

def render_projects():
    """Render the projects section"""
    
    # Load image safely
    image_base64 = safe_load_image(
        IMAGES["projects"],
        DEFAULT_IMAGES["projects"]
    )
    
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
            padding: 20px;
            width: 100%;
        }}

        .description5 {{
            font-family: 'Roboto Condensed', sans-serif;
            font-weight: 600;
            font-size: 18px;
            color: rgba(18,53,36);
            text-align: left;
            line-height: 1.6;
            width: 100%;
            margin: auto;
            padding: 20px;
        }}
        </style>

        <div class="box-contain">
            <div class="heading5">PROJECTS AND EXPERIENCE</div>
            <div class="description5">• ADVANCED REGRESSION MODEL FOR PREDICTING HOUSE PRICES</div>
            <div class="description5">• CLASSIFICATION MODEL FOR PREDICTING CHRONIC HEART DISEASE</div>
            <div class="description5">• PERFORMED K-MEANS CLUSTERING ON NGO DATA TO IDENTIFY COUNTRIES IN GREATEST NEED OF AID, ENABLING MORE EFFECTIVE RESOURCE ALLOCATION</div>
            <div class="description5">• CONDUCTED CASE - CONTROL ANALYSIS ON GSM GENE EXPRESSION DATASET OF PANCREATIC CANCER TO IDENTIFY DIFFERENTIAL GENE EXPRESSION PATTERNS</div>    
            <div class="description5">• MINOR PROJECTS - DATA PREPROCESSING, CLEANING AND PCA ON VARIOUS DATASETS</div>
        </div>
    """, unsafe_allow_html=True)


