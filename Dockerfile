FROM python:3.9-slim

RUN apt-get update -y && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN pip install -e .

EXPOSE 7860

CMD ["python3", "app.py"]
