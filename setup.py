from setuptools import setup, find_packages

setup(
    name="square_skill_api",
    version="0.0.9",
    description="",
    url="www.informatik.tu-darmstadt.de/ukp",
    author="UKP",
    author_email="baumgaertner@ukp.informatik.tu-darmstadt.de",
    packages=find_packages(exclude=("tests", ".gitignore", "pytest.ini", "requirements.dev.txt")),
    dependency_links=['https://github.com/UKP-SQuARE/square-auth.git@v0.0.2#egg=square-auth'],
    install_requires=[
        "uvicorn>=0.15.0",
        "fastapi>=0.65.2",
        "pydantic>=1.8.2",
    ],
)
