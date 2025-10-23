"""
RAMMD Setup Configuration
Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="rammd",
    version="1.0.0",
    author="Agniprabha Chakraborty, Anindya Jana, Manideep Das",
    author_email="agniprabhac.power.ug@jadavpuruniversity.in",
    description="RAMMD: Regime-Adaptive Multimodal Market Decoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RAMMD",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Core dependencies
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "sentencepiece>=0.1.99",
        "accelerate>=0.20.0",
        
        # Time series
        "tsai>=0.3.0",
        "sktime>=0.20.0",
        "statsmodels>=0.14.0",
        
        # Wavelets
        "PyWavelets>=1.4.0",
        
        # Graph Neural Networks
        "torch-geometric>=2.3.0",
        "torch-scatter>=2.1.0",
        "torch-sparse>=0.6.0",
        
        # Financial data
        "yfinance>=0.2.0",
        "pandas-datareader>=0.10.0",
        "alpha-vantage>=2.3.0",
        "finnhub-python>=2.4.0",
        
        # Technical analysis
        "ta>=0.10.0",
        "pandas-ta>=0.3.0",
        
        # Data processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        
        # Drift detection
        "river>=0.18.0",
        
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        
        # Explainability
        "shap>=0.42.0",
        
        # Clustering
        "hmmlearn>=0.3.0",
        
        # Optimization
        "optuna>=3.0.0",
        
        # Logging & monitoring
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        
        # Utils
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    
    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ]
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "rammd-train=scripts.train:main",
            "rammd-evaluate=scripts.evaluate:main",
            "rammd-inference=scripts.inference:main",
        ],
    },
    
    # Package classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    
    # Keywords for discovery
    keywords=[
        "deep-learning",
        "financial-prediction",
        "multimodal-learning",
        "regime-detection",
        "mixture-of-experts",
        "contrastive-learning",
        "graph-neural-networks",
        "wavelet-analysis",
        "stock-prediction",
        "market-forecasting",
    ],
    
    # Include package data
    include_package_data=True,
    
    # Zip safe
    zip_safe=False,
)
