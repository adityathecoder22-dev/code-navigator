from fastapi import FastAPI, HTTPException

from .models import CreateOrderRequest, Order
from .service import OrderService

app = FastAPI(title="order-service")
service = OrderService()


@app.post("/orders", response_model=Order, status_code=201)
def create_order(req: CreateOrderRequest):
    """Create a new order (called by api-gateway)."""
    return service.place_order(req)


@app.get("/orders/{order_id}")
def get_order(order_id: str):
    row = service.repo.get(order_id)
    if row is None:
        raise HTTPException(status_code=404, detail="order not found")
    return row


@app.post("/orders/{order_id}/cancel", status_code=202)
def cancel_order(order_id: str):
    try:
        service.cancel_order(order_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="order not found")
    return {"status": "CANCELLING"}
