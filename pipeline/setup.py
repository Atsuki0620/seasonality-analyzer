"""Setup script for seasonality-analyzer package."""

from setuptools import setup, find_packages

setup(
    name="seasonality-analyzer",
    version="0.1.0",
    description="CLI tool for detecting seasonality patterns in time series data",
    author="Seasonality Analyzer Team",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "ruptures>=1.1.8",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "click>=8.1.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "PyYAML>=6.0.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
        "astropy>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.0.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "python-dotenv>=1.0.0",
        ],
        "full": [
            "prophet>=1.1.4",
            "plotly>=5.15.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "openpyxl>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "seasonality=seasonality.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
