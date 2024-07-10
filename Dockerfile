FROM python:latest

WORKDIR /app

COPY . .

RUN chmod -R 777 /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN playwright install
RUN playwright install-deps

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
