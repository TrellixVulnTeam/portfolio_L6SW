FROM python:3.10

COPY . /mlflow
WORKDIR /mlflow

# Install python packages
RUN pip install -r requirements.txt --upgrade pip

#  RUN python src/models/train_model.py --learning-rate 0.3 --max-depth 5