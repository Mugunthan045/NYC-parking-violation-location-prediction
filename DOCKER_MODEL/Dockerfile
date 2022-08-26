FROM ubuntu:latest
USER root
RUN apt-get update -y
RUN apt-get install openjdk-8-jdk-headless -qq > /dev/null
RUN apt-get install build-essential -y
RUN apt-get install zip -y
RUN apt-get install wget -y
RUN apt-get install python3 -y 
RUN apt-get install python3-pip -y

ENV JAVA_HOME='/usr/lib/jvm/java-8-openjdk-amd64'
ENV PYSPARK_MAJOR_PYTHON_VERSION=3
WORKDIR /opt/application
RUN wget -q https://dlcdn.apache.org/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz
RUN tar xf spark-3.3.0-bin-hadoop3.tgz
ENV SPARK_HOME='/opt/application/spark-3.3.0-bin-hadoop3'
ENV PATH="$PATH:/opt/application/spark-3.3.0-bin-hadoop3/bin"

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY src/ ./src/
COPY Makefile .
COPY main.py .
COPY sample_test.csv ./parking.csv
COPY RFmodel_copy/ ./RFmodel_copy/
RUN make -f Makefile
