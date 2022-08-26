import csv
from kafka import KafkaProducer
import time
from json import dumps



KAFKA_TOPIC = "kafka-spark"
KAFKA_SERVER = "localhost:9092"
CSV_PATH='./sample_processed_test.csv'

csv_data = csv.DictReader(open(CSV_PATH))
post_data = []

for i in csv_data:
    post_data.append(i)
    
if __name__ == "__main__":
    kafka_producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER, value_serializer=lambda m: dumps(m).encode('ascii'))
    for i in post_data:
        print(i)
        kafka_producer.send(topic=KAFKA_TOPIC, key=dumps(i).encode("utf-8"), value=i)
        time.sleep(5)