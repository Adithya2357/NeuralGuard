# NeuralGuard: AI-Powered Intrusion Detection System

NeuralGuard is an AI-powered, real-time intrusion detection system that performs threat analysis and logs potential threats. It leverages Kafka for data streaming, scikit-learn for machine learning-based threat detection, Elasticsearch for logging, and Grafana for visualization. The system is designed to detect anomalous network traffic and alert on potential security threats in real time.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Example Output](#example-output)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

NeuralGuard is designed to provide real-time network security monitoring by detecting potential threats in network traffic. The system is implemented using:

- **Kafka**: For real-time data ingestion and streaming.
- **Elasticsearch**: For storing threat logs and making data searchable.
- **scikit-learn**: For implementing a lightweight machine learning model to classify network traffic.
- **Grafana**: For visualizing detected threats and analyzing trends over time.

## Features

- Real-time threat detection on network traffic.
- Logging of detected threats to Elasticsearch.
- Visualization of network activity and threat trends in Grafana.
- Configurable threat threshold for alerts.

---

## Project Architecture

The main components of NeuralGuard are as follows:

1. **Data Ingestion**: A Kafka producer (`kafka_producer.py`) generates and sends network traffic data to Kafka.
2. **Real-Time Detection**: A Kafka consumer (`detection_consumer.py`) receives data from Kafka, processes it through a trained scikit-learn model, and flags potential threats.
3. **Logging**: Threats detected by the model are logged to Elasticsearch for storage and analysis.
4. **Visualization**: Grafana retrieves data from Elasticsearch and visualizes real-time network activity.

---

## Setup Instructions

### Prerequisites

- **Python 3.10**
- **Kafka** (with Zookeeper)
- **Elasticsearch**
- **Grafana**
- **Virtual Environment** for Python dependencies

### Installation

1. **Clone the Repository**:
   
   ```
   git clone https://github.com/Adithya235/NeuralGuard.git
   cd NeuralGuard
   
   ```
2. **Set Up a Virtual Environment**:
   
   ```
   python3.10 -m venv neuralguard-env
   source neuralguard-env/bin/activate

   ```
3. **Install Required Python Libraries**:
   
   ```
   pip install -r requirements.txt
4. **Start Kafka and Zookeeper**:
   
   -> Zookeeper:
   
   ```
       /opt/kafka/bin/zookeeper-server-start.sh /opt/kafka/config/zookeeper.properties &
   
   ```
   -> Kafka Broker:
   
   ```
   /opt/kafka/bin/kafka-server-start.sh /opt/kafka/config/server.properties &
   
   ```
 5. Start Elasticsearch:
    
   ```
     sudo systemctl start elasticsearch

   ```
 6. Start Grafana:
    
   ```
     sudo systemctl start grafana-server

   ```

 ### Running the Project
 

   Run Kafka Producer (Data Ingestion):
   
    
    python3.10 data_ingestion/kafka_producer.py
    

   Run Kafka Consumer (Real-Time Detection):
   

    python3.10 data_ingestion/kafka_producer.py
    

   View Real-Time Visualization in Grafana:
   

    Open Grafana in your browser: http://localhost:3000
    Use the created dashboards to monitor threat detection and traffic patterns
    

### Example Output


  After running the consumer, you should see output like this in the terminal:
  

    Received message: {'source_ip': '192.168.1.10', 'destination_ip': '10.0.0.3', 'protocol': 'TCP', 'flags': 0}
    Threat score: 0.72
    Threat Detected!
    Logged to Elasticsearch: {'timestamp': '2024-11-03T15:30:00Z', 'source_ip': '192.168.1.10', 'destination_ip': '10.0.0.3', 'protocol': 'TCP', 'flags': 0, 'threat_score': 0.72}


### Future Enhancements


  Advanced Threat Models: Implement more sophisticated models like LSTM for sequential threat detection.
  
  Real-World Data Integration: Integrate real network traffic data for improved model robustness.
  
  Security: Enable authentication for Elasticsearch, Kafka, and Grafana for secure deployment

  
### Contributing


  Contributions are welcome! Please fork the repository, create a branch, and submit a pull request.
  

### License

  This project is licensed under the MIT License. 


   
