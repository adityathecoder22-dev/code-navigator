package payment

import (
	"encoding/json"
	"net/http"

	"github.com/example/sample-shop/payment-service/internal/events"
)

type ChargeRequest struct {
	OrderID   string  `json:"orderId"`
	Amount    float64 `json:"amount"`
	CardToken string  `json:"cardToken"`
}

type ChargeResponse struct {
	PaymentID string `json:"paymentId"`
	Status    string `json:"status"`
	Reason    string `json:"reason,omitempty"`
}

// Handler exposes the HTTP API of payment-service.
type Handler struct {
	gateway     *ProviderGateway
	publisher   *events.Publisher
	idempotency *IdempotencyStore
}

func NewHandler(g *ProviderGateway, p *events.Publisher, s *IdempotencyStore) *Handler {
	return &Handler{gateway: g, publisher: p, idempotency: s}
}

// Charge charges a card for an order. Requests are idempotent per orderId (stored in Redis),
// so a retry from order-service never charges the customer twice.
func (h *Handler) Charge(w http.ResponseWriter, r *http.Request) {
	var req ChargeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	if prev, ok := h.idempotency.Get(r.Context(), req.OrderID); ok {
		writeJSON(w, http.StatusOK, prev)
		return
	}

	result, err := h.gateway.ChargeWithRetry(r.Context(), req.CardToken, req.Amount)
	resp := ChargeResponse{PaymentID: result.ID, Status: "SUCCEEDED"}
	if err != nil {
		resp = ChargeResponse{Status: "FAILED", Reason: err.Error()}
	}
	h.idempotency.Put(r.Context(), req.OrderID, resp)
	h.publisher.PaymentProcessed(req.OrderID, resp.Status, req.Amount)
	writeJSON(w, http.StatusOK, resp)
}

// Refund returns money for a cancelled order.
func (h *Handler) Refund(w http.ResponseWriter, r *http.Request) {
	var req ChargeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	if err := h.gateway.Refund(r.Context(), req.OrderID, req.Amount); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	h.publisher.PaymentProcessed(req.OrderID, "REFUNDED", req.Amount)
	writeJSON(w, http.StatusOK, map[string]string{"status": "REFUNDED"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
