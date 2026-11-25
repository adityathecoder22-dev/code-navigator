package com.shop.inventory.model;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;

@Entity
public class Product {
    @Id
    private String id;
    private String name;
    private double price;
    private int available;
    private int reserved;

    public String getId() { return id; }
    public String getName() { return name; }
    public double getPrice() { return price; }
    public int getAvailable() { return available; }
    public void setAvailable(int available) { this.available = available; }
    public int getReserved() { return reserved; }
    public void setReserved(int reserved) { this.reserved = reserved; }
}
