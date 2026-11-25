package payment

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"
)

var ErrCircuitOpen = errors.New("payment provider circuit breaker is open")

type ProviderResult struct {
	ID string `json:"id"`
}

// ProviderGateway talks to the external card payment provider. It retries transient
// failures with exponential backoff and opens a circuit breaker after repeated failures.
type ProviderGateway struct {
	baseURL  string
	client   *http.Client
	mu       sync.Mutex
	failures int
	openedAt time.Time
}

const (
	maxRetries       = 3
	failureThreshold = 5
	openDuration     = 30 * time.Second
)

func NewProviderGateway(baseURL string, timeout time.Duration) *ProviderGateway {
	return &ProviderGateway{baseURL: baseURL, client: &http.Client{Timeout: timeout}}
}

// ChargeWithRetry calls the provider up to maxRetries times. While the circuit breaker
// is open it fails fast with ErrCircuitOpen instead of calling the provider.
func (g *ProviderGateway) ChargeWithRetry(ctx context.Context, cardToken string, amount float64) (ProviderResult, error) {
	if g.isOpen() {
		return ProviderResult{}, ErrCircuitOpen
	}
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		res, err := g.charge(ctx, cardToken, amount)
		if err == nil {
			g.recordSuccess()
			return res, nil
		}
		lastErr = err
		time.Sleep(time.Duration(200*(1<<attempt)) * time.Millisecond)
	}
	g.recordFailure()
	return ProviderResult{}, fmt.Errorf("charge failed after %d attempts: %w", maxRetries, lastErr)
}

func (g *ProviderGateway) charge(ctx context.Context, cardToken string, amount float64) (ProviderResult, error) {
	body, _ := json.Marshal(map[string]any{"source": cardToken, "amount": amount})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, g.baseURL+"/v1/charges", bytes.NewReader(body))
	resp, err := g.client.Do(req)
	if err != nil {
		return ProviderResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 500 {
		return ProviderResult{}, fmt.Errorf("provider error %d", resp.StatusCode)
	}
	var out ProviderResult
	return out, json.NewDecoder(resp.Body).Decode(&out)
}

// Refund reverses a previous charge.
func (g *ProviderGateway) Refund(ctx context.Context, orderID string, amount float64) error {
	body, _ := json.Marshal(map[string]any{"order": orderID, "amount": amount})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, g.baseURL+"/v1/refunds", bytes.NewReader(body))
	resp, err := g.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}

func (g *ProviderGateway) isOpen() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.failures >= failureThreshold && time.Since(g.openedAt) < openDuration {
		return true
	}
	if g.failures >= failureThreshold {
		g.failures = 0 // half-open: allow a trial request
	}
	return false
}

func (g *ProviderGateway) recordFailure() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.failures++
	if g.failures >= failureThreshold {
		g.openedAt = time.Now()
	}
}

func (g *ProviderGateway) recordSuccess() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.failures = 0
}
