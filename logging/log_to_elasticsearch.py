from elasticsearch import Elasticsearch
from datetime import datetime

# Connect to Elasticsearch (adding the `scheme` parameter)
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

def log_threat(data, threat_score):
    """
    Logs threat data to Elasticsearch.
    
    Args:
        data (dict): Dictionary containing network traffic details.
        threat_score (float): The threat score calculated by the model.
    """
    log_entry = {
        'timestamp': datetime.now(),
        'source_ip': data['source_ip'],
        'destination_ip': data['destination_ip'],
        'protocol': data['protocol'],
        'flags': data['flags'],
        'threat_score': threat_score
    }
    es.index(index='threat-detection', body=log_entry)
    print("Logged to Elasticsearch:", log_entry)
