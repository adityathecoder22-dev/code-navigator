"""Order orchestration: the saga that coordinates inventory, payment and events."""
import uuid

from .clients import InventoryClient, PaymentClient, ServiceUnavailable
from .events import OrderEventPublisher
from .models import CreateOrderRequest, Order, OrderStatus
from .repository import OrderRepository


class OrderService:
    def __init__(self):
        self.inventory = InventoryClient()
        self.payments = PaymentClient()
        self.events = OrderEventPublisher()
        self.repo = OrderRepository()

    def place_order(self, req: CreateOrderRequest) -> Order:
        """Place an order.

        Steps: reserve stock for every item, charge the card, persist the order and publish
        ORDER_CONFIRMED. If payment fails or payment-service is down, previously reserved stock
        is released (compensation) and the order is stored as PAYMENT_FAILED.
        """
        order = Order(
            id=str(uuid.uuid4()), customer_id=req.customerId, email=req.email, items=req.items,
            total=sum(i.quantity * i.unit_price for i in req.items), status=OrderStatus.PENDING,
        )

        reserved = []
        for item in req.items:
            if not self.inventory.reserve(item.product_id, item.quantity, order.id):
                self._release(reserved, order.id)
                order.status = OrderStatus.OUT_OF_STOCK
                self.repo.save(order)
                return order
            reserved.append(item)

        try:
            payment = self.payments.charge(order.id, order.total, req.card_token)
            if payment.get("status") != "SUCCEEDED":
                raise ServiceUnavailable(payment.get("reason", "payment declined"))
            order.payment_id = payment["paymentId"]
            order.status = OrderStatus.CONFIRMED
        except ServiceUnavailable:
            self._release(reserved, order.id)
            order.status = OrderStatus.PAYMENT_FAILED

        self.repo.save(order)
        event = "ORDER_CONFIRMED" if order.status == OrderStatus.CONFIRMED else "ORDER_FAILED"
        self.events.publish(event, order.dict())
        return order

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order and notify other services through the order-events topic."""
        row = self.repo.get(order_id)
        if row is None:
            raise KeyError(order_id)
        self.events.publish("ORDER_CANCELLED", {"id": order_id})

    def _release(self, items, order_id: str) -> None:
        for item in items:
            self.inventory.release(item.product_id, item.quantity, order_id)
