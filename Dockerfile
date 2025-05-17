# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN pip install --upgrade pip

COPY . .

# RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=app

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:create_app()"]