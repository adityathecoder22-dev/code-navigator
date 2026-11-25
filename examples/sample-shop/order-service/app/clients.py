"""HTTP clients for the services order-service depends on."""
import os
import time

import httpx

PAYMENT_SERVICE_URL = os.getenv("PAYMENT_SERVICE_URL", "http://payment-service:9000")
INVENTORY_SERVICE_URL = os.getenv("INVENTORY_SERVICE_URL", "http://inventory-service:8081")


class ServiceUnavailable(Exception):
    """Raised when a downstream service cannot be reached after all retries."""


def _with_retry(fn, attempts: int = 3, backoff: float = 0.5):
    """Retry a call on network errors and 5xx responses with exponential backoff."""
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            last_exc = exc
            time.sleep(backoff * (2 ** attempt))
    raise ServiceUnavailable(str(last_exc))


class InventoryClient:
    """Reserves and releases stock in inventory-service."""

    def __init__(self, base_url: str = INVENTORY_SERVICE_URL):
        self.http = httpx.Client(base_url=base_url, timeout=5.0)

    def reserve(self, product_id: str, quantity: int, order_id: str) -> bool:
        def call():
            r = self.http.post(f"/products/{product_id}/reserve", json={"quantity": quantity, "orderId": order_id})
            if r.status_code == 409:
                return False  # not enough stock
            r.raise_for_status()
            return True
        return _with_retry(call)

    def release(self, product_id: str, quantity: int, order_id: str) -> None:
        def call():
            r = self.http.post(f"/products/{product_id}/release", json={"quantity": quantity, "orderId": order_id})
            r.raise_for_status()
        _with_retry(call)


class PaymentClient:
    """Charges customers through payment-service."""

    def __init__(self, base_url: str = PAYMENT_SERVICE_URL):
        self.http = httpx.Client(base_url=base_url, timeout=10.0)

    def charge(self, order_id: str, amount: float, card_token: str) -> dict:
        def call():
            r = self.http.post("/charge", json={"orderId": order_id, "amount": amount, "cardToken": card_token})
            r.raise_for_status()
            return r.json()
        return _with_retry(call, attempts=2)
