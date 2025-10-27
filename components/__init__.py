# Portfolio Components Package
# This package contains modular components for the portfolio application

from .header import render_header
from .professional_devotion import render_professional_devotion
from .education import render_education
from .certifications import render_certifications
from .skills import render_skills

__all__ = [
    'render_header',
    'render_professional_devotion', 
    'render_education',
    'render_certifications',
    'render_skills'
]


