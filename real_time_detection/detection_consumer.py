from kafka import KafkaConsumer
import json
import numpy as np
import joblib
from log_to_elasticsearch import log_threat  # Importing the logging function

print("Loading model...")  # Debugging line
model = joblib.load('/home/kali/Downloads/NeuralGuard/model/saved_models/threat_detection_model.pkl')
print("Model loaded successfully.")  # Debugging line

# Set up the Kafka consumer
print("Setting up Kafka consumer...")  # Debugging line
consumer = KafkaConsumer(
    'network-traffic',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)
print("Kafka consumer set up. Waiting for messages...")  # Debugging line

THREAT_THRESHOLD = 0.5

def process_data(data):
    protocol_encoded = 1 if data['protocol'] == 'TCP' else 0
    input_data = np.array([[protocol_encoded, data['flags']]])
    return input_data

def detect_threats():
    for message in consumer:
        print("Received message:", message.value)  # Debugging line
        data = message.value
        input_data = process_data(data)
        threat_score = model.predict_proba(input_data)[0][1]  # Probability of being a threat

        print(f"Threat score: {threat_score}")  # Debugging line

        if threat_score >= THREAT_THRESHOLD:
            print("Threat Detected!")
            log_threat(data, threat_score)  # Log to Elasticsearch if it's a threat

if __name__ == "__main__":
    detect_threats()
