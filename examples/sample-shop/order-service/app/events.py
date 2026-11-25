"""Publishes order lifecycle events to Kafka."""
import json
import os

from kafka import KafkaProducer

ORDER_EVENTS_TOPIC = "order-events"


class OrderEventPublisher:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "kafka:9092"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def publish(self, event_type: str, order: dict) -> None:
        """Send an event such as ORDER_CONFIRMED or ORDER_CANCELLED to the order-events topic."""
        self.producer.send("order-events", {"type": event_type, "order": order})
        self.producer.flush()
