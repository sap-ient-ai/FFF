from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    reqs = [line.strip() for line in f.readlines()]

setup(
    name="FFF",
    version="0.1.0",
    author="sap-ient-ai",
    description="FastFeedForward Networks",
    packages=find_packages(),
    install_requires=reqs,
)
