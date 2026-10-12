FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /workspace
COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-dev.txt
COPY . .
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
