import time
import json
import random
from datetime import datetime
import paho.mqtt.client as mqtt

# ---------------------------
# Config (edit if needed)
# ---------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "sic/dibimbing/492/alexander/day7/sensor"
# TOPIC_OUTPUT = "sic/dibimbing/492/alexander/day7/output"
# MODEL_PATH = "iot_temp_model.pkl"   # put the .pkl in same repo

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)


def generateTime():
    print("Reading current time...")
    curr_time = datetime.now().strftime("%H:%M:%S")
    return curr_time

curr_lumen = random.randint(1000,3000)

def generateLumen():
    print("Getting lumen data...")
    global curr_lumen
    change = random.randint(-100, 100)
    curr_lumen += change
    lumen = max(0, min(4500, curr_lumen))
    return lumen

while True:
    print("Generating sensor data...")
    p_time = generateTime()
    p_lumen = generateLumen()
    payload = json.dumps({
        "time": p_time,
        "lumen": p_lumen
    })
    client.publish(TOPIC_SENSOR, payload)
    print("Successfully sent payload: ", payload)

    time.sleep(5)