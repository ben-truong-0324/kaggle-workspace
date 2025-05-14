FROM jupyter/datascience-notebook:latest
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt