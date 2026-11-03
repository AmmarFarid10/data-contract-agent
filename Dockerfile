FROM python:3.12-slim
WORKDIR /srv
RUN pip install --no-cache-dir uv==0.13.4
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
COPY examples ./examples
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --no-editable && useradd --uid 10001 --create-home app && mkdir /data /models && chown app /data /models
ENV PATH="/srv/.venv/bin:$PATH" DATA_PATH=/data/app.sqlite3 MODEL_CACHE=/models
USER app
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
