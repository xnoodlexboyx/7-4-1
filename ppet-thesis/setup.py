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
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "ppet-experiment = scripts.main_experiment:main",
        ],
    },
)