from setuptools import setup, find_packages

setup(
    name="square_skill_api",
    version="0.0.12",
    description="",
    url="www.informatik.tu-darmstadt.de/ukp",
    author="UKP",
    author_email="baumgaertner@ukp.informatik.tu-darmstadt.de",
    packages=find_packages(exclude=("tests", ".gitignore", "pytest.ini", "requirements.dev.txt")),
    install_requires=[
        "square-auth>=0.0.3",
        "uvicorn>=0.15.0",
        "fastapi>=0.65.2",
        "pydantic>=1.8.2",
    ],
)
