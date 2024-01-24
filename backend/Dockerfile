FROM python:3.10.13


WORKDIR /app

ADD requirements.txt /app
RUN pip install --no-cache-dir --upgrade pip && \ 
pip install --no-cache-dir  -r requirements.txt

ADD . /app/

EXPOSE 5000

CMD [ "python", "app.py" ]
