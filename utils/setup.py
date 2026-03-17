from setuptools import setup, find_packages

setup(
    name="autisme-diffusion-ia",
    version="1.0.0",
    description="Application IA pour le catalogue Autisme Diffusion",
    author="Votre Nom",
    author_email="votre.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.17.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
)