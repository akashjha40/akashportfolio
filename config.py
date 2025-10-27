import os
from pathlib import Path

# Base directory configuration
BASE_DIR = Path(__file__).parent.parent
PORTFOLIO_DIR = BASE_DIR / "Portfolio"
ASSETS_DIR = PORTFOLIO_DIR / "assets"

# Image paths - using relative paths from the portfolio directory
IMAGES = {
    "background": ASSETS_DIR / "background.jpg",
    "professional_devotion": ASSETS_DIR / "professional_devotion.jpg",
    "education": ASSETS_DIR / "education.jpg",
    "certifications": ASSETS_DIR / "certifications.jpg",
    "projects": ASSETS_DIR / "projects.jpg"
}

# Data file paths
DATA_FILES = {
    "house_prices": ASSETS_DIR / "Predicted1.csv",
    "ngo_clustering": ASSETS_DIR / "clust_slit.csv",
    "country_data": ASSETS_DIR / "Country-data.csv",
    "gsm_data": ASSETS_DIR / "gsm_Final_2.csv"
}

# Model file paths
MODELS = {
    "heart_disease": ASSETS_DIR / "logist.pkl"
}

# Certificate paths
CERTIFICATES = {
    "cert1": ASSETS_DIR / "cert1.png",
    "cert2": ASSETS_DIR / "cert2.png"
}

# Default fallback images (you can add these to your assets folder)
DEFAULT_IMAGES = {
    "background": "assets/background.jpg",
    "professional_devotion": "assets/professional_devotion.jpg",
    "education": "assets/educations.jpg",
    "certifications": "assets/education.jpg",
    "projects": "assets/projects.jpg"
}
