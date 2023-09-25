FROM --platform=$BUILDPLATFORM python:3.11.4 AS builder

LABEL maintainer="Celso Silvestre <geral@azordev.pt>"

WORKDIR /app

COPY requirements.txt /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT ["python3"]
CMD ["app.py"]

FROM builder as dev-envs

COPY --from=gloursdocker/docker / /
