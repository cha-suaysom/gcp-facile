FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04
#FROM python:3.7
WORKDIR /app
COPY . .

# RUN apt-get update && apt-get install -y --no-install-recommends \
#          wget \
#          curl \
#          libgomp1 \
#          python3-dev && \
#      rm -rf /var/lib/apt/lists/*

# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python3 get-pip.py && \
#     pip install setuptools && \
#     rm get-pip.py

# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#         libgtk2.0-0 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/compat/:$LD_LIBRARY_PATH

RUN pip install pandas==0.24.2
RUN pip install grpcio
RUN pip install tensorflow-gpu==1.13.1
RUN pip install keras==2.2.4
RUN pip install --upgrade google-api-python-client 
RUN pip install --upgrade oauth2client

CMD ["python3", "./server.py"]
