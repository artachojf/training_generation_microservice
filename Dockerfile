FROM python:3.9-slim-buster

WORKDIR /main

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

ENV PORT 8080

CMD ["python", "main.py"]