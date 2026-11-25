"""Consumes order and payment events and emails customers."""
import json
import os
import smtplib
from email.message import EmailMessage

from kafka import KafkaConsumer

TEMPLATES = {
    "ORDER_CONFIRMED": "Your order {id} is confirmed. Total: {total}",
    "ORDER_FAILED": "We could not process your order {id}. You have not been charged.",
    "ORDER_CANCELLED": "Your order {id} has been cancelled.",
    "PAYMENT_REFUNDED": "Your refund for order {orderId} is on its way.",
}


def send_email(to: str, subject: str, body: str) -> None:
    """Send a plain-text email through the SMTP relay configured in SMTP_HOST."""
    msg = EmailMessage()
    msg["From"] = "shop@example.com"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(os.getenv("SMTP_HOST", "mailhog"), 1025) as smtp:
        smtp.send_message(msg)


def handle_event(event: dict) -> None:
    """Render the template for the event type and email the customer, ignoring unknown events."""
    template = TEMPLATES.get(event.get("type"))
    if not template:
        return
    order = event.get("order", event)
    email = order.get("email")
    if email:
        send_email(email, f"Shop update: {event['type']}", template.format(**order))


def main() -> None:
    consumer = KafkaConsumer(
        "order-events",
        "payment-events",
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "kafka:9092"),
        group_id="notification-service",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    for message in consumer:
        handle_event(message.value)


if __name__ == "__main__":
    main()
