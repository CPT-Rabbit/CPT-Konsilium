FROM node:22-slim AS claude-cli
RUN npm install -g @anthropic-ai/claude-code

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd -u 10000 -m konsilium
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ocrmypdf \
        tesseract-ocr-deu \
        tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=claude-cli /usr/local/bin/node /usr/local/bin/node
COPY --from=claude-cli /usr/local/bin/claude /usr/local/bin/claude
COPY --from=claude-cli /usr/local/lib/node_modules /usr/local/lib/node_modules

COPY pyproject.toml README.md DEPLOY.md ./
COPY docs ./docs
COPY roles ./roles
COPY konsilium ./konsilium

RUN pip install --no-cache-dir . \
    && mkdir -p /config /memory /secrets /auth \
    && chown -R 10000:10000 /memory /auth

ENV PIP_NO_INDEX=1
USER 10000:10000

VOLUME ["/config", "/memory", "/secrets", "/auth"]
ENTRYPOINT ["python", "-m", "konsilium"]
CMD ["--help"]
