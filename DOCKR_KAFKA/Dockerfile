FROM ubuntu:latest
USER root

RUN apt-get update -y 
RUN apt-get install curl -y 
RUN apt-get install openjdk-8-jdk-headless -qq > /dev/null
RUN apt-get install python3 python3-pip -y
RUN pip3 install --upgrade pip

RUN mkdir /opt/application
RUN cd /opt/application && curl -O https://dlcdn.apache.org/kafka/3.2.0/kafka_2.13-3.2.0.tgz
RUN cd /opt/application && tar -zxvf kafka_2.13-3.2.0.tgz && rm kafka_2.13-3.2.0.tgz
RUN cd /opt/application && mv kafka_2.13-3.2.0 kafka

ENV JAVA_HOME='/usr/lib/jvm/java-8-openjdk-amd64'

WORKDIR /opt/application

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY kafka_producer.py .
COPY sample_processed_test.csv .
RUN rm /opt/application/kafka/config/server.properties
COPY server.properties /opt/application/kafka/config
COPY start.sh .
CMD ["./start.sh"]

