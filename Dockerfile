# FROM jupyter/datascience-notebook:latest
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade -r requirements.txt

FROM jupyter/datascience-notebook:latest
RUN pip install poetry
RUN poetry config virtualenvs.create false
WORKDIR /home/jovyan/workspace
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi
COPY . .