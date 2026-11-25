package com.shop.inventory.service;

import com.shop.inventory.model.Product;
import com.shop.inventory.repository.ProductRepository;
import java.util.List;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Business logic for stock management. Reservations decrement available stock inside a
 * database transaction with a pessimistic lock so two orders cannot oversell a product.
 */
@Service
public class InventoryService {

    private final ProductRepository repository;

    public InventoryService(ProductRepository repository) {
        this.repository = repository;
    }

    public List<Product> findAll() {
        return repository.findAll();
    }

    public Product findById(String id) {
        return repository.findById(id).orElseThrow(() -> new IllegalArgumentException("Unknown product " + id));
    }

    @Transactional
    public void reserve(String productId, int quantity, String orderId) {
        Product product = repository.findByIdForUpdate(productId)
                .orElseThrow(() -> new IllegalArgumentException("Unknown product " + productId));
        if (product.getAvailable() < quantity) {
            throw new InsufficientStockException(productId, quantity, product.getAvailable());
        }
        product.setAvailable(product.getAvailable() - quantity);
        product.setReserved(product.getReserved() + quantity);
        repository.save(product);
    }

    @Transactional
    public void release(String productId, int quantity, String orderId) {
        Product product = repository.findByIdForUpdate(productId).orElseThrow();
        product.setAvailable(product.getAvailable() + quantity);
        product.setReserved(Math.max(0, product.getReserved() - quantity));
        repository.save(product);
    }
}
