package main

import (
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/example/sample-shop/payment-service/internal/events"
	"github.com/example/sample-shop/payment-service/internal/payment"
)

func main() {
	timeoutMs, _ := strconv.Atoi(getenv("PAYMENT_TIMEOUT_MS", "3000"))
	gateway := payment.NewProviderGateway(os.Getenv("PAYMENT_PROVIDER_URL"), time.Duration(timeoutMs)*time.Millisecond)
	publisher := events.NewPublisher(getenv("KAFKA_BROKERS", "kafka:9092"))
	idempotency := payment.NewIdempotencyStore(getenv("REDIS_ADDR", "redis:6379"))

	handler := payment.NewHandler(gateway, publisher, idempotency)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /charge", handler.Charge)
	mux.HandleFunc("POST /refund", handler.Refund)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })

	log.Println("payment-service listening on :9000")
	log.Fatal(http.ListenAndServe(":9000", mux))
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
