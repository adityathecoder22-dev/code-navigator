package com.shop.inventory.controller;

import com.shop.inventory.model.Product;
import com.shop.inventory.service.InsufficientStockException;
import com.shop.inventory.service.InventoryService;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/products")
public class ProductController {

    private final InventoryService inventoryService;

    public ProductController(InventoryService inventoryService) {
        this.inventoryService = inventoryService;
    }

    /** List all products in the catalogue. */
    @GetMapping
    public List<Product> list() {
        return inventoryService.findAll();
    }

    /** Get a single product with its available stock. */
    @GetMapping("/{id}")
    public Product get(@PathVariable String id) {
        return inventoryService.findById(id);
    }

    /** Reserve stock for an order. Returns 409 CONFLICT when there is not enough stock. */
    @PostMapping("/{id}/reserve")
    public ResponseEntity<Map<String, Object>> reserve(@PathVariable String id, @RequestBody Map<String, Object> body) {
        int quantity = ((Number) body.get("quantity")).intValue();
        String orderId = (String) body.get("orderId");
        try {
            inventoryService.reserve(id, quantity, orderId);
            return ResponseEntity.ok(Map.of("reserved", quantity));
        } catch (InsufficientStockException e) {
            return ResponseEntity.status(HttpStatus.CONFLICT).body(Map.of("error", e.getMessage()));
        }
    }

    /** Release a reservation (compensation when payment fails or an order is cancelled). */
    @PostMapping("/{id}/release")
    public ResponseEntity<Void> release(@PathVariable String id, @RequestBody Map<String, Object> body) {
        inventoryService.release(id, ((Number) body.get("quantity")).intValue(), (String) body.get("orderId"));
        return ResponseEntity.noContent().build();
    }
}
