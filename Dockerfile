FROM python:3.12

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install fastapi uvicorn numpy

COPY ./app /app

WORKDIR /app

RUN chmod +x start.sh

CMD ["/bin/bash", "-c", "./start.sh"]
