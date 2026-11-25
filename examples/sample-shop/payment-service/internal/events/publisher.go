package events

import (
	"context"
	"encoding/json"
	"time"

	"github.com/segmentio/kafka-go"
)

// Publisher writes payment results to the payment-events Kafka topic.
type Publisher struct {
	writer *kafka.Writer
}

func NewPublisher(brokers string) *Publisher {
	return &Publisher{writer: &kafka.Writer{
		Addr:  kafka.TCP(brokers),
		Topic: "payment-events",
	}}
}

// PaymentProcessed publishes a PAYMENT_<status> event for an order.
func (p *Publisher) PaymentProcessed(orderID, status string, amount float64) {
	payload, _ := json.Marshal(map[string]any{
		"type": "PAYMENT_" + status, "orderId": orderID, "amount": amount, "at": time.Now().UTC(),
	})
	_ = p.writer.WriteMessages(context.Background(), kafka.Message{Key: []byte(orderID), Value: payload})
}
