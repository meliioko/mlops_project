FROM python:3.10.13-alpine3.19


WORKDIR /app

ADD requirements.txt /app

RUN pip install --no-cache-dir  -r requirements.txt

ADD . /app/

EXPOSE 5000

CMD [ "py", "app.py" ]