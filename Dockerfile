FROM python:3.8-slim-buster as langchain-serve-img

RUN pip3 install langchain-serve
RUN pip3 install api

CMD [ "lc-serve", "deploy", "local", "api" ]

FROM python:3.8-slim-buster as pdf-gpt-img

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "app.py" ]
