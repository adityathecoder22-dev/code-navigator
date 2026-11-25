package payment

import (
	"context"
	"encoding/json"
	"time"

	"github.com/redis/go-redis/v9"
)

// IdempotencyStore remembers the result of each charge per order in Redis for 24 hours.
type IdempotencyStore struct {
	rdb *redis.Client
}

func NewIdempotencyStore(addr string) *IdempotencyStore {
	return &IdempotencyStore{rdb: redis.NewClient(&redis.Options{Addr: addr})}
}

func (s *IdempotencyStore) Get(ctx context.Context, orderID string) (ChargeResponse, bool) {
	var resp ChargeResponse
	raw, err := s.rdb.Get(ctx, "charge:"+orderID).Bytes()
	if err != nil {
		return resp, false
	}
	return resp, json.Unmarshal(raw, &resp) == nil
}

func (s *IdempotencyStore) Put(ctx context.Context, orderID string, resp ChargeResponse) {
	raw, _ := json.Marshal(resp)
	s.rdb.Set(ctx, "charge:"+orderID, raw, 24*time.Hour)
}
