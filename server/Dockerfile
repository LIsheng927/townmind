FROM python:3.12-slim

# uv：和本地开发用同一套依赖锁文件，保证镜像里的版本和本地一致
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY townmind ./townmind

# NPC 记忆写在 /data，运行时挂一个卷进来，容器重启后记忆还在
ENV TOWNMIND_DATA_DIR=/data \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "townmind.main:app", "--host", "0.0.0.0", "--port", "8000"]
