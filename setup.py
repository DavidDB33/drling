from setuptools import setup
import glob

with open('requirements.txt') as f:
    install_requires = f.read().split()

setup(
    name="drling",
    version="0.0.1",
    scripts=glob.glob("scripts/drling_*.py"),
    install_requires=install_requires
)
