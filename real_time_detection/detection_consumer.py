from kafka import KafkaConsumer
import json
import numpy as np
import joblib
from log_to_elasticsearch import log_threat

print("Loading model...") 
model = joblib.load('/home/kali/Downloads/NeuralGuard/model/saved_models/threat_detection_model.pkl')
print("Model loaded successfully.")  
print("Setting up Kafka consumer...") 
def deserialize_message(x): 
    string_value = x.decode('utf-8') 
    json_value = json.loads(string_value) 
    return json_value
consumer = KafkaConsumer(
    'network-traffic',
    bootstrap_servers='localhost:9092',
    value_deserializer= deserialize_message
)
print("Kafka consumer set up. Waiting for messages...")  

THREAT_THRESHOLD = 0.5

def process_data(data):
    protocol_encoded = 1 if data['protocol'] == 'TCP' else 0
    input_data = np.array([[protocol_encoded, data['flags']]])
    return input_data

def detect_threats():
    for message in consumer:
        print("Received message:", message.value) 
        data = message.value
        input_data = process_data(data)
        threat_score = model.predict_proba(input_data)[0][1]  

        print(f"Threat score: {threat_score}")  
        if threat_score >= THREAT_THRESHOLD:
            print("Threat Detected!")
            log_threat(data, threat_score) 

if __name__ == "__main__":
    detect_threats()
