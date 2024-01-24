from kafka import KafkaConsumer
import json
import pymongo
from model import ResNetModel, inference
import torch
import os
from PIL import Image

# Kafka configuration
KAFKA_BROKER_URL = 'localhost:9092'
KAFKA_TOPIC = 'Human_action'
BATCH_SIZE = 16

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Modify with your MongoDB connection string
db = client["human-action"]  # Replace with your database name

# Check if 'results' collection exists
collection_names = db.list_collection_names()
if 'results' not in collection_names:
    db.create_collection('results')


path = 'model.pth'
model = ResNetModel(15)
model.load_state_dict(torch.load(path))


# Set up Kafka consumer
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER_URL],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest'  # Starts from the earliest message
)

UPLOAD_FOLDER = 'uploads'
def process_messages(messages):
    """
    Process a batch of messages: perform inference and save results to MongoDB.
    """
    for message in messages:
        filename = message['filename']
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        # Perform inference (replace with actual model inference)
        return Image.open(filepath)
# Main loop to process messages in batches
while True:
    batch = []
    for message in consumer:
        batch.append(message.value)
        if len(batch) >= BATCH_SIZE:
            lst = inference(batch, model)
            for i in range(len(lst)):
                data = {
                        "filename": batch[i],
                        "prediction": lst[i]
                        }
                db.results.insert_one(data)
            batch = []  # Reset batch

            
