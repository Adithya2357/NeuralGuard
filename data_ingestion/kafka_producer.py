from kafka import KafkaProducer
import json
import time
import random

def serialize(value):
    return json.dumps(value).encode('utf-8')

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=serialize
)

def generate_traffic_data():
  
    ips = ['192.168.1.10', '192.168.1.20', '10.0.0.3', '10.0.0.8']
    protocols = ['TCP', 'UDP']
    flags = [0, 1] 
    
    return {
        'source_ip': random.choice(ips),
        'destination_ip': random.choice(ips),
        'protocol': random.choice(protocols),
        'flags': random.choice(flags)
    }


def send_traffic():
    while True:
        entry = generate_traffic_data()
        producer.send('network-traffic', entry)
        print(f"Sent data: {entry}")
        time.sleep(1)  # Send data every 1 second

if __name__ == "__main__":
    send_traffic()
