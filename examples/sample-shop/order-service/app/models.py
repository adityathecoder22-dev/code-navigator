from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    OUT_OF_STOCK = "OUT_OF_STOCK"
    CANCELLED = "CANCELLED"


class OrderItem(BaseModel):
    product_id: str
    quantity: int
    unit_price: float


class CreateOrderRequest(BaseModel):
    customerId: str
    email: str
    items: List[OrderItem]
    card_token: str


class Order(BaseModel):
    id: str
    customer_id: str
    email: str
    items: List[OrderItem]
    total: float
    status: OrderStatus
    payment_id: Optional[str] = None
