from setuptools import setup, find_packages
from typing import List

Hypen_e_dot = "-e ."

def get_requirements(file_path:str)->list[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements=[req.replace("\n", "") for req in requirements]

        if Hypen_e_dot in requirements:
            requirements.remove(Hypen_e_dot)

    return requirements

setup(
    name="Student Performance Prediction",
    version="0.0.1",
    author="Yomna Salama",
    author_email="yomna.salama11@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
