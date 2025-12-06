import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_OUTPUT = "sic/dibimbing/492/alexander/day7/output"


def on_connect(client, userdata, flags, rc):
    print("Connected with result code: ", rc)
    client.subscribe(TOPIC_OUTPUT)
    print(f"Subscribed to: {TOPIC_OUTPUT}")

def on_message(client, userdata, msg):
    print(f"Received message: " + msg.payload.decode())

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)

print("Listening for messages...")
client.loop_forever()
