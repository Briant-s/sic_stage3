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
TOPIC_OUTPUT = "sic/dibimbing/492/alexander/day7/output"
# MODEL_PATH = "iot_temp_model.pkl"   # put the .pkl in same repo


ready = False

def on_connect(client, userdata, flags, rc):
    global ready
    print("Connected with result code: ", rc)
    client.subscribe(TOPIC_OUTPUT)
    print(f"Subscribed to: {TOPIC_OUTPUT}")
    ready = True

def on_message(client, userdata, msg):
    print(f"Received message: " + msg.payload.decode())


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)



def generateTime():
    curr_time = datetime.now().strftime("%H:%M:%S")
    return curr_time

curr_lumen = random.randint(1000,3000)

def generateLumen():
    global curr_lumen
    change = random.randint(-500, 500)
    curr_lumen += change
    lumen = max(0, min(4500, curr_lumen))
    return lumen

client.loop_start()

while ready is False:
    print("Waiting to subscribe to output topic...")
    time.sleep(1)

print("Ready! Starting to publish dummy data...")

while True and ready:
    p_time = generateTime()
    p_lumen = generateLumen()
    payload = json.dumps({
        "time": p_time,
        "lumen": p_lumen
    })
    client.publish(TOPIC_SENSOR, payload)
    print("Successfully sent payload: ", payload)

    
    time.sleep(5)