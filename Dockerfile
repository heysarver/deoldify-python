FROM python:3.10-slim-buster

WORKDIR /app

ADD app.py /app/
ADD requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./app.py"]
