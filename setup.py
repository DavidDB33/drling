from setuptools import setup
import glob

with open('requirements.txt') as f:
    install_requires = [
        "gym==0.17.1",
        "matplotlib==3.2.1",
        "numpy==1.18.2",
        "pandas==1.0.3",
        "tensorflow>=2.0.2",
        "tqdm==4.45.0",
        "PyYAML==5.3.1",
    ]

setup(
    name="drling",
    version="0.0.3",
    scripts=glob.glob("scripts/drling_*.py"),
    install_requires=install_requires
)
