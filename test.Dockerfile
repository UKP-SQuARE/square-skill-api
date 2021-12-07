FROM python:3.7.6-slim-buster as base

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY requirements.dev.txt requirements.dev.txt 
RUN pip install -r requirements.dev.txt

COPY square_skill_api square_skill_api

COPY tests tests

RUN python -m pytest --junitxml=test-reports/junit.xml --cov --cov-report=xml:test-reports/coverage.xml --cov-report=html:test-reports/coverage.html]
