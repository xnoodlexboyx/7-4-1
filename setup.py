from setuptools import setup, find_packages

setup(
    name="ppet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
    ],
    extras_require={
        "full": [
            "seaborn",
            "plotly>=5.0.0",
            "pandas",
            "kaleido",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock",
            "pytest-timeout",
            "pytest-xdist",
            "pytest-randomly",
        ],
    },
    entry_points={
        "console_scripts": [
            "ppet-experiment = scripts.main_experiment:main",
            "ppet-visualize = ppet.visualization:main",
            "ppet-bit-analysis = ppet.bit_analysis:main",
            "ppet-statistical-plots = ppet.statistical_plots:main",
            "ppet-defense-dashboard = ppet.defense_dashboard:main",
        ],
    },
    python_requires=">=3.8",
    author="PPET Development Team",
    description="Physical Unclonable Function Emulation and Analysis Framework",
    long_description="A defense-oriented PUF emulation framework for security evaluation under environmental stress conditions.",
    keywords="PUF, security, cryptography, hardware security, defense",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Security :: Cryptography",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)