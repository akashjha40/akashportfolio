import streamlit as st

def render_skills():
    """Render the skills section"""
    
    skills = [
        ("Data Cleaning", "https://img.icons8.com/color/48/000000/data-configuration.png"),
        ("Programming in Python", "https://img.icons8.com/color/48/000000/python.png"),
        ("Machine Learning", "https://img.icons8.com/color/48/000000/artificial-intelligence.png"),
        ("Scikit-learn", "https://img.icons8.com/color/48/000000/artificial-intelligence.png"),
        ("Statsmodels","https://img.icons8.com/color/48/for-experienced.png"),
        ("Pandas", "https://img.icons8.com/color/48/000000/pandas.png"),
        ("MySQL", "https://img.icons8.com/color/48/000000/mysql-logo.png"),
        ("Matplotlib & Seaborn", "https://img.icons8.com/color/48/000000/combo-chart.png"),
        ("MS Excel, PowerPoint, Word", "https://img.icons8.com/color/48/000000/microsoft-office-2019.png"),
        ("Predictive Analysis", "https://img.icons8.com/color/48/000000/line-chart.png"),
        ("Data Storytelling", "https://img.icons8.com/color/48/000000/storytelling.png"),
        ("Streamlit Model Deployment", "https://img.icons8.com/?size=100&id=Rffi8qeb2fK5&format=png&color=000000"),
        ("Fluent in English and Hindi", "https://img.icons8.com/color/48/language-skill.png"),
        ("Strong Communication Skills","https://img.icons8.com/color/48/communication-skills.png" ),
        ("Analytical Thinking","https://img.icons8.com/external-others-bomsymbols-/91/external-analytical-big-data-bluetone-others-bomsymbols-.png")
    ]
    
    # Skills header
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Raleway:wght@100..900&display=swap');

        .skills-box {
            background: rgba(253,253,253,0.5);
            padding: 5px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            width: 20%;
            margin: auto;
            text-align: center;
        }

        .skills-heading {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 30px;
            font-weight: 600;
            text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
            color: rgba(35,43,43);
            margin-bottom: 10px;
            text-align: center;
            margin-top: 14px;          
        }

        .skill-item {
            font-family: 'Raleway', sans-serif;
            font-size: 18px;
            color: black;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-bottom: 15px;
            background: rgba(253,253,253,0.2);
            padding: 5px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            width: 80%;
            margin: auto;
            margin-top: 5%;          
            text-align: center;
        }

        .skill-logo {
            width: 40px;
            height: 40px;
        }
        </style>

        <div class="skills-box">
            <div class="skills-heading">SKILLS</div>
        </div>
    """, unsafe_allow_html=True)

    # Display skills in a grid (3 columns)
    cols = st.columns(3)
    
    for idx, (skill, logo) in enumerate(skills):
        with cols[idx % 3]:
            st.markdown(
                f'<div class="skill-item">'
                f'<img class="skill-logo" src="{logo}"/>'
                f'<span>{skill}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
