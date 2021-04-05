FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn pandas

COPY preprocess.py ./preprocess.py
COPY dataset.csv ./dataset.csv

ENTRYPOINT [ "python", "preprocess.py" ]