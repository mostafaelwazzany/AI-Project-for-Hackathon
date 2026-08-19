FROM node:22-bookworm

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_EXECUTABLE=python3
ENV EMBEDDING_LOCAL_FILES_ONLY=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY web/package*.json ./web/
RUN cd web && npm ci

COPY . .
RUN cd web && npm run build

EXPOSE 3000

CMD ["sh", "-c", "cd web && npx next start -H 0.0.0.0 -p ${PORT:-3000}"]
