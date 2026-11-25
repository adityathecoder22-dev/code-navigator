package com.shop.inventory.service;

public class InsufficientStockException extends RuntimeException {
    public InsufficientStockException(String productId, int requested, int available) {
        super("Product " + productId + ": requested " + requested + ", available " + available);
    }
}
