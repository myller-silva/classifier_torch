FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app

COPY app /app/app
COPY static /app/static
COPY templates /app/templates

ENV FLASK_APP=app
EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:create_app()"]