package com.shop.inventory.events;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.shop.inventory.service.InventoryService;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

/**
 * Listens to order-events. When an order is cancelled, the reserved stock for each item
 * is returned to the available pool.
 */
@Component
public class OrderEventListener {

    private final InventoryService inventoryService;
    private final ObjectMapper mapper = new ObjectMapper();

    public OrderEventListener(InventoryService inventoryService) {
        this.inventoryService = inventoryService;
    }

    @KafkaListener(topics = "order-events", groupId = "inventory-service")
    public void onOrderEvent(String message) throws Exception {
        JsonNode event = mapper.readTree(message);
        if (!"ORDER_CANCELLED".equals(event.path("type").asText())) {
            return;
        }
        JsonNode order = event.path("order");
        for (JsonNode item : order.path("items")) {
            inventoryService.release(item.path("product_id").asText(), item.path("quantity").asInt(), order.path("id").asText());
        }
    }
}
