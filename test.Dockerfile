FROM python:3.7.6-slim-buster as base

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./square_skill_api square_skill_api

COPY requirements.dev.txt requirements.dev.txt 
RUN pip install -r requirements.dev.txt

COPY tests tests

CMD ["python", "-m", "pytest"]
