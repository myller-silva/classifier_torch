FROM python:3.10-slim

RUN pip install --upgrade pip

COPY . .

# RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=app

EXPOSE 5000

# CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:create_app()"]
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
