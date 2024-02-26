FROM python:3.10-slim-buster

WORKDIR /app

ADD app.py /app/
ADD requirements.txt /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-dev \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./app.py"]
