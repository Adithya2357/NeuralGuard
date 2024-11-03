from log_to_elasticsearch import log_threat

# Sample data to log (simulating a network traffic entry)
sample_data = {
    'source_ip': '192.168.1.10',
    'destination_ip': '10.0.0.3',
    'protocol': 'TCP',
    'flags': 1
}
threat_score = 0.8  # Example threat score

# Call the log_threat function to log this sample data
log_threat(sample_data, threat_score)
