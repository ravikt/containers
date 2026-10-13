FROM python:3.12-slim

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt

ENV PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
