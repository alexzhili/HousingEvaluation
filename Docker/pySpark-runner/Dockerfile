FROM python:3.9.7

ENV MASTER "local[*]" 
ENV SPARK_HOME='/spark-3.3.0-bin-hadoop3' 

ENV BASE_URL=https://archive.apache.org/dist/spark
ENV SPARK_VERSION=3.3.0
ENV HADOOP_VERSION=3

RUN cd / \
    && wget ${BASE_URL}/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz  

RUN apt-get -y update && \
    apt-get install --no-install-recommends -y openjdk-11-jre-headless ca-certificates-java unzip wget && \
    apt-get -y autoclean && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/* 

RUN pip --no-cache-dir install --upgrade pyspark nltk mlflow h2o_pysparkling_3.3 hyperopt notebook boto3 matplotlib

ADD ./ /home

WORKDIR /home
