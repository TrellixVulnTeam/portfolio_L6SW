FROM nginx:alpine
RUN rm /etc/nginx/conf.d/default.conf
ADD config/app.conf /etc/nginx/conf.d/default.conf
COPY config/mlflow.conf /etc/nginx/sites-enabled/
RUN mkdir /etc/nginx/html