from setuptools import find_packages, setup

setup(
    name="MedicalChatBox",
    version="0.0.0",
    author="Vaibahv Raj",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "flask",
        "flask-cors",
        "python-dotenv"
    ],
)