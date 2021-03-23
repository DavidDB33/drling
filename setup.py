from setuptools import setup
import glob

install_requires = [
    "PyYAML>=5.3.1",
    "colorama>=0.4.4",
    "gym>=0.17.1",
    "matplotlib>=3.2.1",
    "numpy>=1.18.2",
    "pandas>=1.0.3",
#     "tensorflow>=2.3.1",
    "tqdm>=4.45.0",
]

setup(
    name="drling",
    version="0.0.5",
    description="Deep reinforcement learning algorithms for Researcher's LearnING",
    scripts=glob.glob("scripts/drling_*.py"),
    author_email="david.dominguez@iit.comillas.edu",
    url="https://github.com/DavidDB33/drling",
    install_requires=install_requires,
    packages=["drling"],
)
