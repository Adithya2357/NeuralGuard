from kafka import KafkaProducer
from scapy.all import *  # Import everything from Scapy
import json

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Function to process packets and send to Kafka
def process_packet(packet):
    traffic_data = {
        'protocol': packet.summary(),  # Full packet summary for unknown protocols
        'length': len(packet)
    }

    if IP in packet:  # If the packet has an IP layer
        traffic_data.update({
            'source_ip': packet[IP].src,
            'destination_ip': packet[IP].dst
        })

    elif ARP in packet:  # If the packet is ARP
        traffic_data.update({
            'source_ip': packet[ARP].psrc,
            'destination_ip': packet[ARP].pdst
        })

    else:  # If the packet has no IP or ARP
        traffic_data.update({
            'source_ip': "Unknown",
            'destination_ip': "Unknown"
        })

    # Send data to Kafka
    producer.send('network-traffic', traffic_data)
    print(f"Sent: {traffic_data}")

# Capture live network traffic (all protocols)
def capture_traffic():
    print("Capturing live network traffic... Press Ctrl+C to stop.")
    sniff(prn=process_packet, store=False)  # Capture all packets in real-time

if __name__ == "__main__":
    capture_traffic()
