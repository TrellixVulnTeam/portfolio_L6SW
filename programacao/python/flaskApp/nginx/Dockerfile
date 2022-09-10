FROM nginx:alpine
RUN rm /etc/nginx/conf.d/default.conf
ADD config/app.conf /etc/nginx/conf.d/default.conf
RUN mkdir /etc/nginx/html