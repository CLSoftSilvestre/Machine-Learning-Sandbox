FROM --platform=$BUILDPLATFORM python:3.11.4-slim AS builder

LABEL maintainer="Celso Silvestre <geral@azordev.pt>"

WORKDIR /app

COPY requirements.txt /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY . /app

CMD gunicorn --bind 0.0.0.0:5000 app:app

FROM builder as dev-envs

COPY --from=gloursdocker/docker / /
