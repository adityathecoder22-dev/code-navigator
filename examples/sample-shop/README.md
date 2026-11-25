# Sample Shop

A small polyglot microservice system used to demo and test the Code Intelligence Platform.

| Service | Language | Responsibility |
|---|---|---|
| api-gateway | Node.js / Express | Public REST entry point, JWT auth, routes to internal services |
| order-service | Python / FastAPI | Order lifecycle: reserve stock, charge payment, persist, publish events |
| payment-service | Go | Charges cards through an external provider with retries and a circuit breaker |
| inventory-service | Java / Spring Boot | Product catalogue and stock reservations |
| notification-service | Python | Sends emails when orders and payments change |

## Order flow

1. Client calls `POST /api/orders` on the gateway.
2. Gateway forwards to order-service `POST /orders`.
3. order-service reserves stock in inventory-service (`POST /products/{id}/reserve`).
4. order-service charges the customer via payment-service (`POST /charge`).
5. On success the order is stored as `CONFIRMED` and an `order-events` message is published.
6. If payment fails the reservation is released and the order is marked `PAYMENT_FAILED`.
7. notification-service consumes `order-events` and `payment-events` and emails the customer.
