
# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.7

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . .

# Install production dependencies.
# RUN pip install gunicorn
RUN pip install pandas==0.25.1
RUN pip install grpcio
RUN pip install tensorflow==1.13.1
RUN pip install keras==2.2.4
RUN pip install --upgrade google-api-python-client 
RUN pip install --upgrade oauth2client

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.

#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app

#EXPOSE 50051

CMD ["python", "./server.py"]
