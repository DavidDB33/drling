from setuptools import setup
import glob

setup(
    name="drling",
    version="0.0.1",
    scripts=glob.glob("scripts/drling_*.py")
)
