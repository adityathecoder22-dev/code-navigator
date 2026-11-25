const express = require('express');
const axios = require('axios');

const router = express.Router();
const INVENTORY_SERVICE_URL = process.env.INVENTORY_SERVICE_URL || 'http://inventory-service:8081';

/** List the product catalogue (public, no auth). */
router.get('/', async (req, res) => {
  const response = await axios.get(`${INVENTORY_SERVICE_URL}/products`);
  res.json(response.data);
});

/** Get one product with its current stock level. */
router.get('/:id', async (req, res) => {
  const response = await axios.get(`${INVENTORY_SERVICE_URL}/products/${req.params.id}`);
  res.json(response.data);
});

module.exports = router;
