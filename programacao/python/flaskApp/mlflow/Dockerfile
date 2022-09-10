FROM python:3.10
COPY . /mlflow
WORKDIR /mlflow

RUN pip install -r requirements.txt --upgrade pip


RUN mlflow
  
  
#   command: mlflow server --backend-store-uri ${MLFLOW__SQL_ALCHEMY_CONN} 