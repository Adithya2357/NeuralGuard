from kafka import KafkaProducer
from scapy.all import * 
import json
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
def process_packet(packet):
    traffic_data = {
        'protocol': packet.summary(),  
        'length': len(packet)
    }
    if IP in packet:  
        traffic_data.update({
            'source_ip': packet[IP].src,
            'destination_ip': packet[IP].dst
        })
    elif ARP in packet: 
        traffic_data.update({
            'source_ip': packet[ARP].psrc,
            'destination_ip': packet[ARP].pdst
        })
    else:  
        traffic_data.update({
            'source_ip': "Unknown",
            'destination_ip': "Unknown"
        })
    producer.send('network-traffic', traffic_data)
    print(f"Sent: {traffic_data}")
def capture_traffic():
    print("Capturing live network traffic... Press Ctrl+C to stop.")
    sniff(prn=process_packet, store=False) 
if __name__ == "__main__":
    capture_traffic()
