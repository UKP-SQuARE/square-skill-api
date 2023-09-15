from setuptools import find_packages, setup

__version__ = "0.0.46"

setup(
    name="square_skill_api",
    version=__version__,
    license="MIT",
    description="",
    url="https://github.com/UKP-SQuARE/square-skill-api",
    download_url=f"https://github.com/UKP-SQuARE/square-skill-api/archive/refs/tags/v{__version__}.tar.gz",
    author="UKP",
    author_email="baumgaertner@ukp.informatik.tu-darmstadt.de",
    packages=find_packages(
        exclude=("tests", ".gitignore", "pytest.ini", "requirements.dev.txt")
    ),
    install_requires=[
        "uvicorn>=0.15.0",
        "fastapi>=0.65.2",
        "pydantic>=1.8.2",
        "numpy>=1.21.3",
        "square-elk-json-formatter==0.0.3",
    ],
)
