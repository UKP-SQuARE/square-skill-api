from setuptools import setup

setup(
    name="square_skill_api",
    version="0.0.1",    
    description="",
    url="www.informatik.tu-darmstadt.de/ukp",
    author="UKP",
    author_email="baumgaertner@ukp.informatik.tu-darmstadt.de",
    packages=["square_skill_api"],
    install_requires=[
        "uvicorn>=0.15.0",
        "fastapi>=0.65.2",                     
        "pydantic>=1.8.2",                     
    ],
)
