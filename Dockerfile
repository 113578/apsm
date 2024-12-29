FROM python:3.11.11

WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/

RUN pip install -r requirements.txt

COPY . /usr/src/app

EXPOSE 8000 8501

ENV PYTHONPATH=/usr/src/app

ENTRYPOINT ["python", "apsm/app/main.py"]
