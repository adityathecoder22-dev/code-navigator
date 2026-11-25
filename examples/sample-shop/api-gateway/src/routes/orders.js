const express = require('express');
const axios = require('axios');

const router = express.Router();
const ORDER_SERVICE_URL = process.env.ORDER_SERVICE_URL || 'http://order-service:8000';

/** Create an order for the authenticated user by forwarding to order-service. */
router.post('/', async (req, res) => {
  try {
    const payload = { ...req.body, customerId: req.user.sub, email: req.user.email };
    const response = await axios.post(`${ORDER_SERVICE_URL}/orders`, payload, { timeout: 10000 });
    res.status(response.status).json(response.data);
  } catch (err) {
    const status = err.response ? err.response.status : 502;
    res.status(status).json({ error: 'order creation failed', detail: err.message });
  }
});

/** Fetch a single order. */
router.get('/:id', async (req, res) => {
  try {
    const response = await axios.get(`${ORDER_SERVICE_URL}/orders/${req.params.id}`);
    res.json(response.data);
  } catch (err) {
    res.status(err.response ? err.response.status : 502).json({ error: err.message });
  }
});

module.exports = router;
