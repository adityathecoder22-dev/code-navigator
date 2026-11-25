"""PostgreSQL persistence for orders."""
import json
import os

import psycopg

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://shop:shop@postgres:5432/orders")


class OrderRepository:
    def __init__(self, dsn: str = DATABASE_URL):
        self.conn = psycopg.connect(dsn, autocommit=True)

    def save(self, order) -> None:
        self.conn.execute(
            "INSERT INTO orders (id, customer_id, email, items, total, status, payment_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status, payment_id = EXCLUDED.payment_id",
            (order.id, order.customer_id, order.email, json.dumps([i.dict() for i in order.items]),
             order.total, order.status.value, order.payment_id),
        )

    def get(self, order_id: str):
        row = self.conn.execute("SELECT * FROM orders WHERE id = %s", (order_id,)).fetchone()
        return row
